//! Guest program for zero-knowledge proof generation
//! Implements the Japanese Bankers Association risk assessment logic as described in the paper

#![no_main]

use risc0_zkvm::guest::env;
use serde::{Deserialize, Serialize};
use sha2::{Sha256, Digest};

// Input structure for the risk assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserRiskProfile {
    pub age: u32,
    pub income: u64,
    pub savings: u64,
    pub has_mortgage: bool,
    pub has_retirement_plan: bool,
    pub investment_experience: InvestmentExperience,
    pub risk_questions: [u8; 10], // Responses to 10 financial questions (1-3 scale)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InvestmentExperience {
    Beginner = 1,
    Intermediate = 2,
    Advanced = 3,
}

// Output structure containing the risk category and verification data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAssessmentResult {
    pub risk_category: RiskCategory,
    pub confidence_score: u32,
    pub data_hash: [u8; 32], // SHA-256 hash of input for verification
    pub computation_proof: ComputationProof,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskCategory {
    Conservative = 1,
    SteadyGrowth = 2,
    Balanced = 3,
    AggressiveInvestment = 4,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputationProof {
    pub total_score: u32,
    pub age_factor: u32,
    pub income_factor: u32,
    pub experience_factor: u32,
    pub risk_tolerance_score: u32,
}

risc0_zkvm::guest::entry!(main);

fn main() {
    // Read the input from the host
    let input: UserRiskProfile = env::read();

    // Generate the risk assessment result
    let result = compute_risk_assessment(&input);

    // Commit the result to the journal (this becomes part of the proof)
    env::commit(&result);
}

/// Core risk assessment logic implementing the Japanese Bankers Association methodology
fn compute_risk_assessment(profile: &UserRiskProfile) -> RiskAssessmentResult {
    // Step 1: Hash the input data for integrity verification
    let data_hash = compute_data_hash(profile);

    // Step 2: Calculate various risk factors
    let age_factor = calculate_age_factor(profile.age);
    let income_factor = calculate_income_factor(profile.income, profile.savings);
    let experience_factor = calculate_experience_factor(&profile.investment_experience);
    let risk_tolerance_score = calculate_risk_tolerance(&profile.risk_questions);

    // Step 3: Apply weighting factors based on additional profile data
    let mortgage_adjustment = if profile.has_mortgage { -5 } else { 0 };
    let retirement_adjustment = if profile.has_retirement_plan { 3 } else { -2 };

    // Step 4: Calculate total score
    let total_score = age_factor + income_factor + experience_factor + risk_tolerance_score
        + mortgage_adjustment + retirement_adjustment;

    // Step 5: Categorize based on total score
    let risk_category = categorize_risk(total_score);

    // Step 6: Calculate confidence score based on data completeness and consistency
    let confidence_score = calculate_confidence_score(profile, total_score);

    RiskAssessmentResult {
        risk_category,
        confidence_score,
        data_hash,
        computation_proof: ComputationProof {
            total_score: total_score as u32,
            age_factor,
            income_factor,
            experience_factor,
            risk_tolerance_score,
        },
    }
}

/// Compute SHA-256 hash of the input data for verification
fn compute_data_hash(profile: &UserRiskProfile) -> [u8; 32] {
    let mut hasher = Sha256::new();

    // Create a deterministic string representation of the profile
    let profile_string = format!(
        "age:{},income:{},savings:{},mortgage:{},retirement:{},experience:{:?},questions:{:?}",
        profile.age,
        profile.income,
        profile.savings,
        profile.has_mortgage,
        profile.has_retirement_plan,
        profile.investment_experience,
        profile.risk_questions
    );

    hasher.update(profile_string.as_bytes());
    let result = hasher.finalize();

    let mut hash_array = [0u8; 32];
    hash_array.copy_from_slice(&result);
    hash_array
}

/// Calculate age-based risk factor
/// Younger investors can typically take more risk
fn calculate_age_factor(age: u32) -> u32 {
    match age {
        18..=25 => 15,  // High risk tolerance
        26..=35 => 12,  // Above average risk tolerance
        36..=45 => 8,   // Moderate risk tolerance
        46..=55 => 5,   // Conservative approach
        56..=65 => 2,   // Low risk tolerance
        _ => 0,         // Very conservative
    }
}

/// Calculate income and savings based risk factor
/// Higher income/savings typically allows for more risk
fn calculate_income_factor(income: u64, savings: u64) -> u32 {
    let total_financial_capacity = income + (savings / 10); // Savings weighted less than annual income

    match total_financial_capacity {
        0..=30_000 => 2,
        30_001..=50_000 => 4,
        50_001..=75_000 => 6,
        75_001..=100_000 => 8,
        100_001..=150_000 => 10,
        150_001..=250_000 => 12,
        _ => 15,
    }
}

/// Calculate experience-based risk factor
fn calculate_experience_factor(experience: &InvestmentExperience) -> u32 {
    match experience {
        InvestmentExperience::Beginner => 2,
        InvestmentExperience::Intermediate => 6,
        InvestmentExperience::Advanced => 10,
    }
}

/// Calculate risk tolerance based on questionnaire responses
/// Questions are rated 1-3 where typically:
/// 1 = Conservative/Risk-averse response
/// 2 = Moderate response
/// 3 = Aggressive/Risk-seeking response
fn calculate_risk_tolerance(questions: &[u8; 10]) -> u32 {
    let total: u32 = questions.iter().map(|&q| q as u32).sum();

    // Apply weighting to certain questions that are more indicative of risk tolerance
    let weighted_score =
        (questions[0] as u32) * 2 +  // Investment time horizon (most important)
        (questions[1] as u32) * 2 +  // Risk vs return preference (most important)
        (questions[2] as u32) * 1 +  // Market volatility comfort
        (questions[3] as u32) * 1 +  // Loss tolerance
        (questions[4] as u32) * 2 +  // Investment goals (important)
        (questions[5] as u32) * 1 +  // Portfolio diversification preference
        (questions[6] as u32) * 1 +  // Liquidity needs
        (questions[7] as u32) * 1 +  // Economic outlook influence
        (questions[8] as u32) * 1 +  // Professional advice reliance
        (questions[9] as u32) * 1;   // Past investment behavior

    // Normalize the weighted score to a reasonable range
    (weighted_score * 10) / 15
}

/// Categorize the final risk level based on total score
fn categorize_risk(total_score: i32) -> RiskCategory {
    match total_score {
        0..=25 => RiskCategory::Conservative,
        26..=45 => RiskCategory::SteadyGrowth,
        46..=65 => RiskCategory::Balanced,
        _ => RiskCategory::AggressiveInvestment,
    }
}

/// Calculate confidence score based on data quality and consistency
fn calculate_confidence_score(profile: &UserRiskProfile, total_score: i32) -> u32 {
    let mut confidence = 100u32;

    // Reduce confidence for edge cases or inconsistent data

    // Age consistency checks
    if profile.age < 18 || profile.age > 100 {
        confidence = confidence.saturating_sub(20);
    }

    // Income/savings consistency
    if profile.savings > profile.income * 20 {
        confidence = confidence.saturating_sub(10); // Unusual savings rate
    }

    // Experience vs questionnaire consistency
    let avg_question_score = profile.risk_questions.iter().map(|&q| q as u32).sum::<u32>() / 10;
    let experience_score = match profile.investment_experience {
        InvestmentExperience::Beginner => 1,
        InvestmentExperience::Intermediate => 2,
        InvestmentExperience::Advanced => 3,
    };

    if (avg_question_score as i32 - experience_score as i32).abs() > 1 {
        confidence = confidence.saturating_sub(15); // Experience doesn't match risk tolerance
    }

    // Check for extreme scores
    if total_score < 5 || total_score > 80 {
        confidence = confidence.saturating_sub(10);
    }

    // Ensure all question responses are valid (1-3)
    for &response in &profile.risk_questions {
        if response < 1 || response > 3 {
            confidence = confidence.saturating_sub(25);
            break;
        }
    }

    confidence.max(50) // Minimum confidence of 50%
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conservative_profile() {
        let profile = UserRiskProfile {
            age: 60,
            income: 40_000,
            savings: 50_000,
            has_mortgage: true,
            has_retirement_plan: true,
            investment_experience: InvestmentExperience::Beginner,
            risk_questions: [1, 1, 2, 1, 1, 2, 1, 2, 1, 1],
        };

        let result = compute_risk_assessment(&profile);
        assert!(matches!(result.risk_category, RiskCategory::Conservative));
        assert!(result.confidence_score >= 50);
    }

    #[test]
    fn test_aggressive_profile() {
        let profile = UserRiskProfile {
            age: 28,
            income: 120_000,
            savings: 80_000,
            has_mortgage: false,
            has_retirement_plan: true,
            investment_experience: InvestmentExperience::Advanced,
            risk_questions: [3, 3, 3, 2, 3, 3, 2, 3, 2, 3],
        };

        let result = compute_risk_assessment(&profile);
        assert!(matches!(result.risk_category, RiskCategory::AggressiveInvestment));
        assert!(result.confidence_score >= 70);
    }

    #[test]
    fn test_data_hash_consistency() {
        let profile = UserRiskProfile {
            age: 35,
            income: 75_000,
            savings: 25_000,
            has_mortgage: true,
            has_retirement_plan: false,
            investment_experience: InvestmentExperience::Intermediate,
            risk_questions: [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        };

        let hash1 = compute_data_hash(&profile);
        let hash2 = compute_data_hash(&profile);
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_edge_cases() {
        // Test minimum age
        let young_profile = UserRiskProfile {
            age: 18,
            income: 25_000,
            savings: 1_000,
            has_mortgage: false,
            has_retirement_plan: false,
            investment_experience: InvestmentExperience::Beginner,
            risk_questions: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        };

        let result = compute_risk_assessment(&young_profile);
        assert!(result.confidence_score >= 50);

        // Test high income/savings
        let wealthy_profile = UserRiskProfile {
            age: 45,
            income: 500_000,
            savings: 1_000_000,
            has_mortgage: false,
            has_retirement_plan: true,
            investment_experience: InvestmentExperience::Advanced,
            risk_questions: [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        };

        let result = compute_risk_assessment(&wealthy_profile);
        assert!(matches!(result.risk_category, RiskCategory::AggressiveInvestment));
    }
}
