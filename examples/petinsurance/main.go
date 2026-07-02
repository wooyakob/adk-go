// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package main demonstrates a pet insurance claims processing agent built on
// the ADK framework. It uses a multi-agent architecture:
//   - A root orchestrator routes requests to specialized sub-agents.
//   - The intake agent collects and submits new claims.
//   - The adjudication agent evaluates coverage and approves/denies claims.
//   - The status agent answers questions about existing claims.
package main

import (
	"context"
	"fmt"
	"log"
	"math/rand/v2"
	"os"
	"strings"
	"sync"
	"time"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/full"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/agenttool"
	"google.golang.org/adk/tool/functiontool"
	"google.golang.org/genai"
)

// -----------------------------------------------------------------------------
// Domain types used by tools
// -----------------------------------------------------------------------------

// PolicyType represents the tier of coverage a policy holder has.
type PolicyType string

const (
	PolicyBasic        PolicyType = "basic"
	PolicyComprehensive PolicyType = "comprehensive"
	PolicyPremium      PolicyType = "premium"
)

// ClaimStatus reflects where a claim is in the processing pipeline.
type ClaimStatus string

const (
	StatusPending  ClaimStatus = "pending"
	StatusReview   ClaimStatus = "under_review"
	StatusApproved ClaimStatus = "approved"
	StatusDenied   ClaimStatus = "denied"
	StatusPaid     ClaimStatus = "paid"
)

// claim is an internal record stored in our in-memory "database".
type claim struct {
	ID              string
	PolicyNumber    string
	PetName         string
	Species         string
	Diagnosis       string
	TreatmentDate   string
	VetName         string
	VetClinic       string
	TotalCost       float64
	Status          ClaimStatus
	Reimbursement   float64
	DenialReason    string
	SubmittedAt     time.Time
}

// policyHolder is an in-memory record of a policy.
type policyHolder struct {
	PolicyNumber string
	OwnerName    string
	PetName      string
	Species      string
	Breed        string
	Age          int
	PolicyType   PolicyType
	Deductible   float64
	Copay        float64 // percentage owner pays (e.g. 0.20 = 20%)
	AnnualLimit  float64
	UsedThisYear float64
}

// -----------------------------------------------------------------------------
// In-memory "database" – pre-seeded with demo data
// -----------------------------------------------------------------------------

// dbMu guards concurrent access to the policies and claims maps.
var dbMu sync.RWMutex

var (
	policies = map[string]*policyHolder{
		"POL-1001": {
			PolicyNumber: "POL-1001",
			OwnerName:    "Alice Johnson",
			PetName:      "Biscuit",
			Species:      "dog",
			Breed:        "Golden Retriever",
			Age:          4,
			PolicyType:   PolicyComprehensive,
			Deductible:   200.00,
			Copay:        0.10,
			AnnualLimit:  10000.00,
			UsedThisYear: 450.00,
		},
		"POL-1002": {
			PolicyNumber: "POL-1002",
			OwnerName:    "Bob Martinez",
			PetName:      "Whiskers",
			Species:      "cat",
			Breed:        "Domestic Shorthair",
			Age:          7,
			PolicyType:   PolicyBasic,
			Deductible:   500.00,
			Copay:        0.20,
			AnnualLimit:  5000.00,
			UsedThisYear: 0.00,
		},
		"POL-1003": {
			PolicyNumber: "POL-1003",
			OwnerName:    "Carol White",
			PetName:      "Mango",
			Species:      "dog",
			Breed:        "French Bulldog",
			Age:          2,
			PolicyType:   PolicyPremium,
			Deductible:   100.00,
			Copay:        0.05,
			AnnualLimit:  20000.00,
			UsedThisYear: 1200.00,
		},
	}

	claims = map[string]*claim{
		"CLM-5001": {
			ID:            "CLM-5001",
			PolicyNumber:  "POL-1001",
			PetName:       "Biscuit",
			Species:       "dog",
			Diagnosis:     "Torn ACL – surgical repair",
			TreatmentDate: "2026-06-01",
			VetName:       "Dr. Sarah Nguyen",
			VetClinic:     "Paws & Claws Animal Hospital",
			TotalCost:     3200.00,
			Status:        StatusReview,
			SubmittedAt:   time.Now().AddDate(0, 0, -5),
		},
		"CLM-5002": {
			ID:            "CLM-5002",
			PolicyNumber:  "POL-1002",
			PetName:       "Whiskers",
			Species:       "cat",
			Diagnosis:     "Hyperthyroidism – medication",
			TreatmentDate: "2026-05-28",
			VetName:       "Dr. Mike Chen",
			VetClinic:     "Sunny Side Vets",
			TotalCost:     620.00,
			Status:        StatusApproved,
			Reimbursement: 96.00,
			SubmittedAt:   time.Now().AddDate(0, 0, -10),
		},
	}
)

// -----------------------------------------------------------------------------
// Coverage rules per policy type
// -----------------------------------------------------------------------------

type coverageRule struct {
	covered bool
	notes   string
}

// coveredConditions defines what each policy tier covers.
func checkCoverage(policyType PolicyType, diagnosis string) coverageRule {
	dx := strings.ToLower(diagnosis)

	// Exclusions apply to all tiers.
	exclusions := []string{"grooming", "nail trim", "teeth cleaning", "wellness", "vaccine", "flea", "heartworm prevention"}
	for _, ex := range exclusions {
		if strings.Contains(dx, ex) {
			return coverageRule{covered: false, notes: fmt.Sprintf("'%s' is classified as routine/preventive care and is excluded from all policies.", ex)}
		}
	}

	switch policyType {
	case PolicyBasic:
		// Basic only covers accidents.
		accidents := []string{"fracture", "laceration", "bite", "ingestion", "foreign body", "torn", "broken", "cut", "wound", "swallowed"}
		for _, a := range accidents {
			if strings.Contains(dx, a) {
				return coverageRule{covered: true, notes: "Covered under Basic – accident benefit."}
			}
		}
		return coverageRule{covered: false, notes: "Basic plan covers accidents only. This appears to be an illness/condition not covered under the Basic tier."}
	case PolicyComprehensive:
		// Comprehensive covers accidents + illness.
		return coverageRule{covered: true, notes: "Covered under Comprehensive plan – accidents and illness benefit."}
	case PolicyPremium:
		// Premium covers accidents, illness, and some specialist/hereditary conditions.
		return coverageRule{covered: true, notes: "Covered under Premium plan – accidents, illness, hereditary conditions, and specialist care."}
	}
	return coverageRule{covered: false, notes: "Unknown policy type."}
}

// -----------------------------------------------------------------------------
// Tool definitions
// -----------------------------------------------------------------------------

// --- lookup_policy ---

type LookupPolicyInput struct {
	PolicyNumber string `json:"policy_number" jsonschema:"description=The policy number to look up (e.g. POL-1001)"`
}

type LookupPolicyOutput struct {
	Found        bool    `json:"found"`
	PolicyNumber string  `json:"policy_number,omitempty"`
	OwnerName    string  `json:"owner_name,omitempty"`
	PetName      string  `json:"pet_name,omitempty"`
	Species      string  `json:"species,omitempty"`
	Breed        string  `json:"breed,omitempty"`
	PetAge       int     `json:"pet_age,omitempty"`
	PolicyType   string  `json:"policy_type,omitempty"`
	Deductible   float64 `json:"deductible,omitempty"`
	CopayPercent float64 `json:"copay_percent,omitempty"`
	AnnualLimit  float64 `json:"annual_limit,omitempty"`
	UsedThisYear float64 `json:"used_this_year,omitempty"`
	RemainingBenefit float64 `json:"remaining_benefit,omitempty"`
	Error        string  `json:"error,omitempty"`
}

func lookupPolicy(_ tool.Context, input LookupPolicyInput) (LookupPolicyOutput, error) {
	dbMu.RLock()
	defer dbMu.RUnlock()
	p, ok := policies[input.PolicyNumber]
	if !ok {
		return LookupPolicyOutput{Found: false, Error: fmt.Sprintf("No policy found for number %q.", input.PolicyNumber)}, nil
	}
	return LookupPolicyOutput{
		Found:            true,
		PolicyNumber:     p.PolicyNumber,
		OwnerName:        p.OwnerName,
		PetName:          p.PetName,
		Species:          p.Species,
		Breed:            p.Breed,
		PetAge:           p.Age,
		PolicyType:       string(p.PolicyType),
		Deductible:       p.Deductible,
		CopayPercent:     p.Copay * 100,
		AnnualLimit:      p.AnnualLimit,
		UsedThisYear:     p.UsedThisYear,
		RemainingBenefit: p.AnnualLimit - p.UsedThisYear,
	}, nil
}

// --- check_coverage ---

type CheckCoverageInput struct {
	PolicyNumber string `json:"policy_number" jsonschema:"description=The policy number to check coverage for"`
	Diagnosis    string `json:"diagnosis" jsonschema:"description=The diagnosis or condition/treatment to check"`
}

type CheckCoverageOutput struct {
	PolicyNumber string `json:"policy_number"`
	Diagnosis    string `json:"diagnosis"`
	Covered      bool   `json:"covered"`
	Notes        string `json:"notes"`
}

func checkCoverageTool(_ tool.Context, input CheckCoverageInput) (CheckCoverageOutput, error) {
	dbMu.RLock()
	defer dbMu.RUnlock()
	p, ok := policies[input.PolicyNumber]
	if !ok {
		return CheckCoverageOutput{PolicyNumber: input.PolicyNumber, Diagnosis: input.Diagnosis, Covered: false, Notes: fmt.Sprintf("Policy %q not found.", input.PolicyNumber)}, nil
	}
	rule := checkCoverage(p.PolicyType, input.Diagnosis)
	return CheckCoverageOutput{
		PolicyNumber: input.PolicyNumber,
		Diagnosis:    input.Diagnosis,
		Covered:      rule.covered,
		Notes:        rule.notes,
	}, nil
}

// --- submit_claim ---

type SubmitClaimInput struct {
	PolicyNumber  string  `json:"policy_number" jsonschema:"description=The insured's policy number"`
	PetName       string  `json:"pet_name" jsonschema:"description=Name of the pet being treated"`
	Diagnosis     string  `json:"diagnosis" jsonschema:"description=Diagnosis or description of the condition/treatment"`
	TreatmentDate string  `json:"treatment_date" jsonschema:"description=Date of treatment in YYYY-MM-DD format"`
	VetName       string  `json:"vet_name" jsonschema:"description=Name of the attending veterinarian"`
	VetClinic     string  `json:"vet_clinic" jsonschema:"description=Name of the veterinary clinic"`
	TotalCost     float64 `json:"total_cost" jsonschema:"description=Total invoice amount in USD"`
}

type SubmitClaimOutput struct {
	Success   bool    `json:"success"`
	ClaimID   string  `json:"claim_id,omitempty"`
	Status    string  `json:"status,omitempty"`
	Message   string  `json:"message"`
	EstimatedReimbursement float64 `json:"estimated_reimbursement,omitempty"`
}

func submitClaim(_ tool.Context, input SubmitClaimInput) (SubmitClaimOutput, error) {
	dbMu.Lock()
	defer dbMu.Unlock()

	p, ok := policies[input.PolicyNumber]
	if !ok {
		return SubmitClaimOutput{Success: false, Message: fmt.Sprintf("Policy %q not found. Please verify the policy number.", input.PolicyNumber)}, nil
	}

	// Quick eligibility check.
	rule := checkCoverage(p.PolicyType, input.Diagnosis)
	if !rule.covered {
		return SubmitClaimOutput{
			Success: false,
			Message: fmt.Sprintf("Claim cannot be submitted: %s", rule.notes),
		}, nil
	}

	// Generate a unique claim ID, retrying on collision.
	var id string
	for {
		id = fmt.Sprintf("CLM-%04d", 5000+rand.IntN(9000))
		if _, exists := claims[id]; !exists {
			break
		}
	}

	// Quick reimbursement estimate.
	remaining := p.AnnualLimit - p.UsedThisYear
	if remaining < 0 {
		remaining = 0
	}
	eligible := input.TotalCost - p.Deductible
	if eligible < 0 {
		eligible = 0
	}
	reimbursement := eligible * (1 - p.Copay)
	if reimbursement > remaining {
		reimbursement = remaining
	}

	c := &claim{
		ID:            id,
		PolicyNumber:  input.PolicyNumber,
		PetName:       input.PetName,
		Species:       p.Species,
		Diagnosis:     input.Diagnosis,
		TreatmentDate: input.TreatmentDate,
		VetName:       input.VetName,
		VetClinic:     input.VetClinic,
		TotalCost:     input.TotalCost,
		Status:        StatusPending,
		Reimbursement: reimbursement,
		SubmittedAt:   time.Now(),
	}
	claims[id] = c

	return SubmitClaimOutput{
		Success:                true,
		ClaimID:                id,
		Status:                 string(StatusPending),
		Message:                fmt.Sprintf("Claim %s submitted successfully for %s's treatment at %s. You will receive a decision within 5–7 business days.", id, input.PetName, input.VetClinic),
		EstimatedReimbursement: reimbursement,
	}, nil
}

// --- get_claim_status ---

type GetClaimStatusInput struct {
	ClaimID string `json:"claim_id" jsonschema:"description=The claim ID to look up (e.g. CLM-5001)"`
}

type GetClaimStatusOutput struct {
	Found         bool    `json:"found"`
	ClaimID       string  `json:"claim_id,omitempty"`
	PolicyNumber  string  `json:"policy_number,omitempty"`
	PetName       string  `json:"pet_name,omitempty"`
	Diagnosis     string  `json:"diagnosis,omitempty"`
	TreatmentDate string  `json:"treatment_date,omitempty"`
	TotalCost     float64 `json:"total_cost,omitempty"`
	Status        string  `json:"status,omitempty"`
	Reimbursement float64 `json:"reimbursement,omitempty"`
	DenialReason  string  `json:"denial_reason,omitempty"`
	SubmittedAt   string  `json:"submitted_at,omitempty"`
	Error         string  `json:"error,omitempty"`
}

func getClaimStatus(_ tool.Context, input GetClaimStatusInput) (GetClaimStatusOutput, error) {
	dbMu.RLock()
	defer dbMu.RUnlock()
	c, ok := claims[input.ClaimID]
	if !ok {
		return GetClaimStatusOutput{Found: false, Error: fmt.Sprintf("No claim found with ID %q.", input.ClaimID)}, nil
	}
	return GetClaimStatusOutput{
		Found:         true,
		ClaimID:       c.ID,
		PolicyNumber:  c.PolicyNumber,
		PetName:       c.PetName,
		Diagnosis:     c.Diagnosis,
		TreatmentDate: c.TreatmentDate,
		TotalCost:     c.TotalCost,
		Status:        string(c.Status),
		Reimbursement: c.Reimbursement,
		DenialReason:  c.DenialReason,
		SubmittedAt:   c.SubmittedAt.Format(time.RFC3339),
	}, nil
}

// --- list_claims_for_policy ---

type ListClaimsInput struct {
	PolicyNumber string `json:"policy_number" jsonschema:"description=The policy number whose claims to list"`
}

type ClaimSummary struct {
	ClaimID       string  `json:"claim_id"`
	Diagnosis     string  `json:"diagnosis"`
	TreatmentDate string  `json:"treatment_date"`
	TotalCost     float64 `json:"total_cost"`
	Status        string  `json:"status"`
}

type ListClaimsOutput struct {
	PolicyNumber string         `json:"policy_number"`
	Claims       []ClaimSummary `json:"claims"`
	TotalClaims  int            `json:"total_claims"`
}

func listClaimsForPolicy(_ tool.Context, input ListClaimsInput) (ListClaimsOutput, error) {
	dbMu.RLock()
	defer dbMu.RUnlock()
	var result []ClaimSummary
	for _, c := range claims {
		if c.PolicyNumber == input.PolicyNumber {
			result = append(result, ClaimSummary{
				ClaimID:       c.ID,
				Diagnosis:     c.Diagnosis,
				TreatmentDate: c.TreatmentDate,
				TotalCost:     c.TotalCost,
				Status:        string(c.Status),
			})
		}
	}
	return ListClaimsOutput{
		PolicyNumber: input.PolicyNumber,
		Claims:       result,
		TotalClaims:  len(result),
	}, nil
}

// --- adjudicate_claim ---

type AdjudicateClaimInput struct {
	ClaimID      string `json:"claim_id" jsonschema:"description=The claim ID to adjudicate"`
	Decision     string `json:"decision" jsonschema:"description=Either 'approve' or 'deny'"`
	DenialReason string `json:"denial_reason,omitempty" jsonschema:"description=Required if decision is 'deny'; reason for denial"`
}

type AdjudicateClaimOutput struct {
	Success       bool    `json:"success"`
	ClaimID       string  `json:"claim_id"`
	Decision      string  `json:"decision"`
	Reimbursement float64 `json:"reimbursement,omitempty"`
	Message       string  `json:"message"`
}

func adjudicateClaim(_ tool.Context, input AdjudicateClaimInput) (AdjudicateClaimOutput, error) {
	dbMu.Lock()
	defer dbMu.Unlock()

	c, ok := claims[input.ClaimID]
	if !ok {
		return AdjudicateClaimOutput{Success: false, ClaimID: input.ClaimID, Message: fmt.Sprintf("Claim %q not found.", input.ClaimID)}, nil
	}
	if c.Status == StatusApproved || c.Status == StatusDenied {
		return AdjudicateClaimOutput{Success: false, ClaimID: c.ID, Message: fmt.Sprintf("Claim %s has already been adjudicated (status: %s).", c.ID, c.Status)}, nil
	}

	switch strings.ToLower(input.Decision) {
	case "approve":
		c.Status = StatusApproved
		p := policies[c.PolicyNumber]
		if p != nil {
			p.UsedThisYear += c.Reimbursement
		}
		return AdjudicateClaimOutput{
			Success:       true,
			ClaimID:       c.ID,
			Decision:      "approved",
			Reimbursement: c.Reimbursement,
			Message:       fmt.Sprintf("Claim %s approved. Reimbursement of $%.2f will be processed within 3–5 business days.", c.ID, c.Reimbursement),
		}, nil
	case "deny":
		if input.DenialReason == "" {
			return AdjudicateClaimOutput{Success: false, ClaimID: c.ID, Message: "A denial reason is required when denying a claim."}, nil
		}
		c.Status = StatusDenied
		c.DenialReason = input.DenialReason
		return AdjudicateClaimOutput{
			Success:  true,
			ClaimID:  c.ID,
			Decision: "denied",
			Message:  fmt.Sprintf("Claim %s denied. Reason: %s. The policyholder will be notified.", c.ID, input.DenialReason),
		}, nil
	default:
		return AdjudicateClaimOutput{Success: false, ClaimID: c.ID, Message: fmt.Sprintf("Invalid decision %q. Must be 'approve' or 'deny'.", input.Decision)}, nil
	}
}

// --- calculate_reimbursement ---

type CalcReimbursementInput struct {
	PolicyNumber string  `json:"policy_number" jsonschema:"description=Policy number to use for calculation"`
	TotalCost    float64 `json:"total_cost" jsonschema:"description=Total invoice amount in USD"`
}

type CalcReimbursementOutput struct {
	PolicyNumber        string  `json:"policy_number"`
	TotalCost           float64 `json:"total_cost"`
	Deductible          float64 `json:"deductible"`
	EligibleAmount      float64 `json:"eligible_amount"`
	CopayPercent        float64 `json:"copay_percent"`
	InsurancePays       float64 `json:"insurance_pays"`
	OwnerPays           float64 `json:"owner_pays"`
	RemainingBenefit    float64 `json:"remaining_annual_benefit"`
	Notes               string  `json:"notes"`
}

func calculateReimbursement(_ tool.Context, input CalcReimbursementInput) (CalcReimbursementOutput, error) {
	dbMu.RLock()
	defer dbMu.RUnlock()

	p, ok := policies[input.PolicyNumber]
	if !ok {
		return CalcReimbursementOutput{}, fmt.Errorf("policy %q not found", input.PolicyNumber)
	}

	remaining := p.AnnualLimit - p.UsedThisYear
	if remaining < 0 {
		remaining = 0
	}
	eligible := input.TotalCost - p.Deductible
	if eligible < 0 {
		eligible = 0
	}
	insurancePays := eligible * (1 - p.Copay)
	if insurancePays > remaining {
		insurancePays = remaining
	}
	ownerPays := input.TotalCost - insurancePays

	notes := fmt.Sprintf("Based on %s policy: $%.2f deductible applied, then %.0f%% copay on remaining eligible amount.", p.PolicyType, p.Deductible, p.Copay*100)
	if insurancePays == remaining && remaining < eligible*(1-p.Copay) {
		notes += fmt.Sprintf(" Note: annual benefit cap of $%.2f reached.", p.AnnualLimit)
	}

	return CalcReimbursementOutput{
		PolicyNumber:     p.PolicyNumber,
		TotalCost:        input.TotalCost,
		Deductible:       p.Deductible,
		EligibleAmount:   eligible,
		CopayPercent:     p.Copay * 100,
		InsurancePays:    insurancePays,
		OwnerPays:        ownerPays,
		RemainingBenefit: remaining - insurancePays,
		Notes:            notes,
	}, nil
}

// -----------------------------------------------------------------------------
// Helper: build a functiontool or fatal
// -----------------------------------------------------------------------------

func mustTool[TArgs, TResults any](cfg functiontool.Config, handler functiontool.Func[TArgs, TResults]) tool.Tool {
	t, err := functiontool.New(cfg, handler)
	if err != nil {
		log.Fatalf("failed to create tool %q: %v", cfg.Name, err)
	}
	return t
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

func main() {
	ctx := context.Background()

	model, err := gemini.NewModel(ctx, "gemini-2.5-flash", &genai.ClientConfig{
		APIKey: os.Getenv("GOOGLE_API_KEY"),
	})
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	// -- Tools --

	lookupPolicyTool := mustTool(functiontool.Config{
		Name:        "lookup_policy",
		Description: "Look up a pet insurance policy by policy number. Returns owner details, pet info, coverage tier, deductible, copay, annual limit, and remaining benefit.",
	}, lookupPolicy)

	checkCoverageToolFn := mustTool(functiontool.Config{
		Name:        "check_coverage",
		Description: "Check whether a specific diagnosis or treatment is covered under a given policy. Returns a covered boolean and an explanation.",
	}, checkCoverageTool)

	submitClaimTool := mustTool(functiontool.Config{
		Name:        "submit_claim",
		Description: "Submit a new pet insurance claim. Requires the policy number, pet name, diagnosis, treatment date, vet name, clinic name, and total invoice cost. Returns a claim ID and estimated reimbursement.",
	}, submitClaim)

	getClaimStatusTool := mustTool(functiontool.Config{
		Name:        "get_claim_status",
		Description: "Get the current status of a submitted claim by claim ID. Returns status, reimbursement amount, and denial reason if applicable.",
	}, getClaimStatus)

	listClaimsTool := mustTool(functiontool.Config{
		Name:        "list_claims_for_policy",
		Description: "List all claims filed under a specific policy number, with their statuses and treatment dates.",
	}, listClaimsForPolicy)

	adjudicateClaimTool := mustTool(functiontool.Config{
		Name:        "adjudicate_claim",
		Description: "Approve or deny a claim after review. Requires a claim ID, a decision ('approve' or 'deny'), and a denial reason if denying. Updates the claim record and notifies the policyholder.",
	}, adjudicateClaim)

	calcReimbursementTool := mustTool(functiontool.Config{
		Name:        "calculate_reimbursement",
		Description: "Calculate the estimated reimbursement for a given cost and policy, showing deductible, eligible amount, copay, and what both the insurer and owner will pay.",
	}, calculateReimbursement)

	// -- Sub-agents --

	// intakeAgent handles new claim submissions.
	intakeAgent, err := llmagent.New(llmagent.Config{
		Name:  "intake_agent",
		Model: model,
		Description: "Handles the end-to-end intake of new pet insurance claims. " +
			"Collects required information from the policyholder, verifies coverage, calculates expected reimbursement, and submits the claim.",
		Instruction: `You are a compassionate and efficient Pet Insurance Claims Intake Specialist.

Your job is to help policyholders submit new insurance claims for their pets' veterinary treatment.

When a policyholder wants to file a claim:
1. Ask for their policy number if not provided, then use lookup_policy to verify it.
2. Confirm the pet's name and species match what's on the policy.
3. Use check_coverage to verify the diagnosis/treatment is covered before proceeding.
4. If not covered, explain why clearly and empathetically — do not submit the claim.
5. Collect all required information: diagnosis, treatment date (YYYY-MM-DD), vet name, clinic name, and total invoice amount.
6. Use calculate_reimbursement to show the policyholder an estimate before submitting.
7. Confirm the details with the policyholder, then call submit_claim.
8. Provide the claim ID and next steps.

Be warm, professional, and thorough. Pet owners are often stressed about their pet's health — acknowledge that with empathy.
Always clearly explain what is and is not covered, and why.`,
		Tools: []tool.Tool{
			lookupPolicyTool,
			checkCoverageToolFn,
			calcReimbursementTool,
			submitClaimTool,
		},
	})
	if err != nil {
		log.Fatalf("Failed to create intake agent: %v", err)
	}

	// adjudicationAgent reviews and decides on existing claims.
	adjudicationAgent, err := llmagent.New(llmagent.Config{
		Name:  "adjudication_agent",
		Model: model,
		Description: "Reviews pending claims, evaluates coverage eligibility, and issues approve/deny decisions with full justification.",
		Instruction: `You are a Pet Insurance Claims Adjudicator responsible for reviewing submitted claims and making fair, policy-compliant decisions.

When asked to review or adjudicate a claim:
1. Use get_claim_status to retrieve the full claim details.
2. Use lookup_policy to review the policy terms (coverage tier, deductible, copay, annual limit, remaining benefit).
3. Use check_coverage to verify whether the diagnosis is covered under the policy tier.
4. Use calculate_reimbursement to confirm the correct payout amount.
5. Make a decision:
   - If covered and within limits: call adjudicate_claim with decision="approve".
   - If not covered or excluded: call adjudicate_claim with decision="deny" and a clear, specific denial_reason.
6. Summarize your decision to the user, including the reimbursement amount (if approved) or the denial reason (if denied).

Be objective and fair. Always base decisions strictly on the policy terms.
When denying, be clear but kind — the owner may appeal.`,
		Tools: []tool.Tool{
			getClaimStatusTool,
			lookupPolicyTool,
			checkCoverageToolFn,
			calcReimbursementTool,
			adjudicateClaimTool,
		},
	})
	if err != nil {
		log.Fatalf("Failed to create adjudication agent: %v", err)
	}

	// statusAgent answers questions about existing claims and policy details.
	statusAgent, err := llmagent.New(llmagent.Config{
		Name:  "status_agent",
		Model: model,
		Description: "Answers questions about existing claims and policy coverage details.",
		Instruction: `You are a Pet Insurance Customer Service Representative specializing in claim status inquiries.

You help policyholders:
- Check the status of a specific claim by claim ID
- View all claims filed under a policy number
- Understand their policy coverage, deductible, and remaining annual benefit
- Understand why a claim was approved or denied

Use the available tools to look up accurate information. Never guess or make up claim or policy details.
Be clear, concise, and empathetic — the policyholder may be worried about their pet.

If a claim is denied, explain the denial reason clearly and mention that the policyholder may contact us to appeal.`,
		Tools: []tool.Tool{
			getClaimStatusTool,
			listClaimsTool,
			lookupPolicyTool,
		},
	})
	if err != nil {
		log.Fatalf("Failed to create status agent: %v", err)
	}

	// -- Root orchestrator --

	rootAgent, err := llmagent.New(llmagent.Config{
		Name:  "pet_insurance_claims_agent",
		Model: model,
		Description: "PawProtect Insurance – AI-powered pet insurance claims assistant. " +
			"Handles new claim submissions, claim status inquiries, coverage questions, and adjudication.",
		Instruction: `You are the PawProtect Insurance virtual claims assistant. You help pet owners and insurance staff with pet insurance claims.

You have three specialist teams available:
- **Intake Team** (intake_agent): For filing NEW claims. Route here when a user wants to submit a claim for a vet visit.
- **Adjudication Team** (adjudication_agent): For REVIEWING and DECIDING on pending claims. Route here when a staff member wants to approve or deny a claim.
- **Status Team** (status_agent): For CHECKING existing claims or policy details. Route here for status updates, claim history, coverage questions, or policy lookups.

Routing rules:
- "I need to file a claim" / "submit a claim" / "my pet was treated" → intake_agent
- "approve this claim" / "review claim CLM-XXXX" / "adjudicate" → adjudication_agent
- "what's the status of my claim" / "check claim" / "my policy details" / "what's covered" → status_agent
- Greetings or unclear requests: briefly introduce yourself and ask what the user needs help with today.

Always greet users warmly. PawProtect cares deeply about pets and their families.
Do not attempt to answer questions outside of pet insurance — politely redirect.`,
		Tools: []tool.Tool{
			agenttool.New(intakeAgent, nil),
			agenttool.New(adjudicationAgent, nil),
			agenttool.New(statusAgent, nil),
		},
	})
	if err != nil {
		log.Fatalf("Failed to create root agent: %v", err)
	}

	config := &launcher.Config{
		AgentLoader: agent.NewSingleLoader(rootAgent),
	}

	l := full.NewLauncher()
	if err = l.Execute(ctx, config, os.Args[1:]); err != nil {
		log.Fatalf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}
