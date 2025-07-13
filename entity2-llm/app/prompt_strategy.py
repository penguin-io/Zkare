#!/usr/bin/env python3
"""
Prompt Strategy Implementation for Entity 2 LLM Advisor

This module implements the prompting strategy described in the paper:
"Generating Privacy-Preserving Personalized Advice with Zero-Knowledge Proofs and LLMs"

The strategy partitions user traits into:
- d0: Unverifiable exploratory traits
- d1: Verifiable traits (proven via ZKP)

And generates contexts with varying emphasis on these traits.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

import structlog
from jinja2 import Template, Environment, FileSystemLoader

from .models import (
    VerifiedTraits, UnverifiableTraits, Domain, RiskCategory,
    AgeBracket, IncomeLevel, ExperienceLevel
)

logger = structlog.get_logger()


class ContextType(str, Enum):
    """Context types for prompt generation"""
    BASELINE = "c0"           # No specific emphasis
    EMPHASIZE_D0 = "c1"       # Emphasize unverifiable traits
    EMPHASIZE_D1 = "c2"       # Emphasize verifiable traits
    MODERATE_D1 = "c3"        # Moderate emphasis on verifiable traits


class PromptStrategy:
    """
    Implementation of the paper's prompting strategy for privacy-preserving advice generation

    The strategy creates personalized and consistent responses by leveraging both
    unverifiable and verifiable user traits with different emphasis levels.
    """

    def __init__(self, config_path: str = "/app/config/prompts.yaml"):
        self.config_path = config_path
        self.config = {}
        self.templates = {}
        self.jinja_env = None

        # Load configuration and templates
        self._load_config()
        self._setup_jinja_environment()
        self._load_templates()

    def _load_config(self) -> None:
        """Load prompt configuration from YAML file"""
        try:
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f)
            else:
                # Use default configuration
                self.config = self._get_default_config()
                logger.warning(f"Config file not found at {self.config_path}, using defaults")

        except Exception as e:
            logger.error(f"Failed to load prompt config: {e}")
            self.config = self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default prompt configuration"""
        return {
            "contexts": {
                "c0": {
                    "name": "baseline",
                    "description": "No specific trait emphasis",
                    "d0_weight": 0.5,
                    "d1_weight": 0.5
                },
                "c1": {
                    "name": "emphasize_unverifiable",
                    "description": "Emphasize unverifiable traits",
                    "d0_weight": 0.8,
                    "d1_weight": 0.2
                },
                "c2": {
                    "name": "emphasize_verifiable",
                    "description": "Emphasize verifiable traits",
                    "d0_weight": 0.2,
                    "d1_weight": 0.8
                },
                "c3": {
                    "name": "moderate_verifiable",
                    "description": "Moderate emphasis on verifiable traits",
                    "d0_weight": 0.3,
                    "d1_weight": 0.7
                }
            },
            "domains": {
                "financial": {
                    "system_prompt": "You are a helpful financial advisor providing personalized investment and financial planning advice.",
                    "emphasis_keywords": ["investment", "risk", "portfolio", "financial goals", "market conditions"]
                },
                "healthcare": {
                    "system_prompt": "You are a knowledgeable healthcare advisor providing general health and wellness guidance.",
                    "emphasis_keywords": ["health", "wellness", "symptoms", "lifestyle", "medical history"]
                },
                "general": {
                    "system_prompt": "You are a helpful advisor providing personalized guidance and recommendations.",
                    "emphasis_keywords": ["preferences", "situation", "goals", "context", "circumstances"]
                }
            },
            "response_format": {
                "max_length": 500,
                "include_reasoning": True,
                "formal_tone": False,
                "structured_response": True
            }
        }

    def _setup_jinja_environment(self) -> None:
        """Setup Jinja2 environment for template rendering"""
        template_dir = Path(self.config_path).parent / "templates"
        if template_dir.exists():
            self.jinja_env = Environment(
                loader=FileSystemLoader(str(template_dir)),
                trim_blocks=True,
                lstrip_blocks=True
            )
        else:
            self.jinja_env = Environment()

    def _load_templates(self) -> None:
        """Load prompt templates"""
        self.templates = {
            "system_prompt": self._get_system_prompt_template(),
            "proposal_prompt": self._get_proposal_prompt_template(),
            "explanation_prompt": self._get_explanation_prompt_template(),
            "chat_prompt": self._get_chat_prompt_template(),
            "context_builder": self._get_context_builder_template()
        }

    def _get_system_prompt_template(self) -> str:
        """Get system prompt template"""
        return """
You are an expert advisor providing personalized recommendations based on user traits and context.

Guidelines:
- Provide accurate, helpful, and personalized advice
- Consider both verified and unverified user characteristics
- Maintain user privacy by not revealing specific personal details
- Be empathetic and understanding of individual circumstances
- Provide actionable recommendations when appropriate

Domain: {{ domain }}
Context Type: {{ context_type }}
Trait Emphasis: {{ trait_emphasis }}
""".strip()

    def _get_proposal_prompt_template(self) -> str:
        """Get proposal generation prompt template"""
        return """
{{ system_prompt }}

User Context:
{{ user_context }}

User Question: {{ query }}

Please provide a personalized recommendation that addresses the user's question while considering their specific circumstances and traits. Focus on providing practical, actionable advice.

Response:
""".strip()

    def _get_explanation_prompt_template(self) -> str:
        """Get explanation generation prompt template"""
        return """
{{ system_prompt }}

User Context:
{{ user_context }}

User Question: {{ query }}

Proposed Answer: {{ proposed_answer }}

Please provide a clear explanation of why this recommendation is appropriate for the user's specific situation. Explain the reasoning behind the advice, considering their verified traits and circumstances.

Explanation:
""".strip()

    def _get_chat_prompt_template(self) -> str:
        """Get chat conversation prompt template"""
        return """
{{ system_prompt }}

User Context:
{{ user_context }}

Conversation History:
{{ conversation_history }}

Current Message: {{ message }}

Please respond naturally while maintaining awareness of the user's context and previous conversation. Be helpful and engaging.

Response:
""".strip()

    def _get_context_builder_template(self) -> str:
        """Get context builder template"""
        return """
{% if verified_traits %}
Verified User Characteristics (High Confidence):
{% if verified_traits.risk_category %}
- Risk Tolerance: {{ verified_traits.risk_category.value.replace('_', ' ').title() }}
{% endif %}
{% if verified_traits.age_bracket %}
- Age Group: {{ verified_traits.age_bracket.value.replace('_', ' ').title() }}
{% endif %}
{% if verified_traits.income_level %}
- Income Level: {{ verified_traits.income_level.value.replace('_', ' ').title() }}
{% endif %}
{% if verified_traits.experience_level %}
- Experience Level: {{ verified_traits.experience_level.value.title() }}
{% endif %}
{% if verified_traits.confidence_score %}
- Assessment Confidence: {{ verified_traits.confidence_score }}%
{% endif %}
{% endif %}

{% if unverifiable_traits %}
Additional Context (User-Provided):
{% if unverifiable_traits.personality_type %}
- Personality: {{ unverifiable_traits.personality_type }}
{% endif %}
{% if unverifiable_traits.communication_style %}
- Communication Style: {{ unverifiable_traits.communication_style }}
{% endif %}
{% if unverifiable_traits.goals %}
- Goals: {{ unverifiable_traits.goals | join(', ') }}
{% endif %}
{% if unverifiable_traits.concerns %}
- Concerns: {{ unverifiable_traits.concerns | join(', ') }}
{% endif %}
{% if unverifiable_traits.additional_context %}
- Additional Context: {{ unverifiable_traits.additional_context }}
{% endif %}
{% endif %}

Context Emphasis: {{ emphasis_description }}
""".strip()

    def create_context(
        self,
        verified_traits: Optional[VerifiedTraits] = None,
        unverifiable_traits: Optional[UnverifiableTraits] = None,
        domain: Optional[Domain] = None,
        context_type: ContextType = ContextType.BASELINE
    ) -> Dict[str, str]:
        """
        Create context for both proposal and explanation generation

        Returns:
            Dictionary with 'proposal' and 'explanation' context strings
        """
        # Get context configuration
        context_config = self.config["contexts"][context_type.value]
        d0_weight = context_config["d0_weight"]
        d1_weight = context_config["d1_weight"]

        # Build emphasis description
        emphasis_description = self._build_emphasis_description(context_type, d0_weight, d1_weight)

        # Generate contexts for proposal and explanation
        proposal_context = self._generate_context_string(
            verified_traits=verified_traits,
            unverifiable_traits=unverifiable_traits,
            domain=domain,
            context_type=context_type,
            emphasis_description=emphasis_description,
            phase="proposal"
        )

        explanation_context = self._generate_context_string(
            verified_traits=verified_traits,
            unverifiable_traits=unverifiable_traits,
            domain=domain,
            context_type=context_type,
            emphasis_description=emphasis_description,
            phase="explanation"
        )

        return {
            "proposal": proposal_context,
            "explanation": explanation_context
        }

    def _build_emphasis_description(
        self,
        context_type: ContextType,
        d0_weight: float,
        d1_weight: float
    ) -> str:
        """Build description of trait emphasis"""
        if context_type == ContextType.BASELINE:
            return "Balanced consideration of all available information"
        elif context_type == ContextType.EMPHASIZE_D0:
            return "Primary focus on user preferences and self-reported traits"
        elif context_type == ContextType.EMPHASIZE_D1:
            return "Primary focus on verified and objective characteristics"
        elif context_type == ContextType.MODERATE_D1:
            return "Moderate emphasis on verified traits with consideration of user preferences"
        else:
            return f"Custom emphasis (Verified: {d1_weight}, Unverified: {d0_weight})"

    def _generate_context_string(
        self,
        verified_traits: Optional[VerifiedTraits],
        unverifiable_traits: Optional[UnverifiableTraits],
        domain: Optional[Domain],
        context_type: ContextType,
        emphasis_description: str,
        phase: str
    ) -> str:
        """Generate context string using template"""
        template = Template(self.templates["context_builder"])

        return template.render(
            verified_traits=verified_traits,
            unverifiable_traits=unverifiable_traits,
            domain=domain.value if domain else "general",
            context_type=context_type.value,
            emphasis_description=emphasis_description,
            phase=phase
        )

    def create_proposal_prompt(
        self,
        query: str,
        context: str,
        domain: Optional[Domain] = None
    ) -> str:
        """Create prompt for generating proposed answer (A_prop)"""
        domain_config = self.config["domains"].get(
            domain.value if domain else "general",
            self.config["domains"]["general"]
        )

        system_prompt = domain_config["system_prompt"]

        template = Template(self.templates["proposal_prompt"])
        return template.render(
            system_prompt=system_prompt,
            user_context=context,
            query=query,
            domain=domain.value if domain else "general"
        )

    def create_explanation_prompt(
        self,
        query: str,
        proposed_answer: str,
        context: str,
        domain: Optional[Domain] = None
    ) -> str:
        """Create prompt for generating explanation (A_exp)"""
        domain_config = self.config["domains"].get(
            domain.value if domain else "general",
            self.config["domains"]["general"]
        )

        system_prompt = domain_config["system_prompt"]

        template = Template(self.templates["explanation_prompt"])
        return template.render(
            system_prompt=system_prompt,
            user_context=context,
            query=query,
            proposed_answer=proposed_answer,
            domain=domain.value if domain else "general"
        )

    def create_chat_prompt(
        self,
        message: str,
        history: List[Dict[str, str]],
        context: str = "",
        domain: Optional[Domain] = None
    ) -> str:
        """Create prompt for chat conversation"""
        domain_config = self.config["domains"].get(
            domain.value if domain else "general",
            self.config["domains"]["general"]
        )

        system_prompt = domain_config["system_prompt"]

        # Format conversation history
        conversation_history = ""
        for turn in history[-10:]:  # Limit to last 10 turns
            role = turn.get("role", "user")
            content = turn.get("content", "")
            conversation_history += f"{role.title()}: {content}\n"

        template = Template(self.templates["chat_prompt"])
        return template.render(
            system_prompt=system_prompt,
            user_context=context,
            conversation_history=conversation_history.strip(),
            message=message,
            domain=domain.value if domain else "general"
        )

    def create_chat_context(self, verified_traits: VerifiedTraits) -> str:
        """Create context for chat conversations from verified traits"""
        context_parts = []

        if verified_traits.risk_category:
            context_parts.append(f"Risk tolerance: {verified_traits.risk_category.value.replace('_', ' ')}")

        if verified_traits.age_bracket:
            context_parts.append(f"Age group: {verified_traits.age_bracket.value.replace('_', ' ')}")

        if verified_traits.experience_level:
            context_parts.append(f"Experience: {verified_traits.experience_level.value}")

        if verified_traits.confidence_score:
            context_parts.append(f"Assessment confidence: {verified_traits.confidence_score}%")

        return "User profile: " + ", ".join(context_parts) if context_parts else ""

    def analyze_trait_consistency(
        self,
        verified_traits: Optional[VerifiedTraits],
        unverifiable_traits: Optional[UnverifiableTraits]
    ) -> Dict[str, Any]:
        """Analyze consistency between verified and unverifiable traits"""
        analysis = {
            "consistency_score": 1.0,
            "conflicts": [],
            "recommendations": []
        }

        if not verified_traits or not unverifiable_traits:
            return analysis

        # Check for potential conflicts
        conflicts = []

        # Example: Check if risk tolerance matches personality
        if (verified_traits.risk_category == RiskCategory.CONSERVATIVE and
            unverifiable_traits.personality_type and
            "aggressive" in unverifiable_traits.personality_type.lower()):
            conflicts.append("Conservative risk profile conflicts with aggressive personality")

        # Example: Check if experience level matches self-reported goals
        if (verified_traits.experience_level == ExperienceLevel.BEGINNER and
            unverifiable_traits.goals and
            any("advanced" in goal.lower() for goal in unverifiable_traits.goals)):
            conflicts.append("Beginner experience level conflicts with advanced goals")

        analysis["conflicts"] = conflicts
        analysis["consistency_score"] = max(0.0, 1.0 - (len(conflicts) * 0.2))

        # Generate recommendations for handling conflicts
        if conflicts:
            analysis["recommendations"] = [
                "Emphasize verified traits for core recommendations",
                "Use unverifiable traits for personalization only",
                "Explain potential discrepancies to user"
            ]

        return analysis

    def generate_multi_context_responses(
        self,
        query: str,
        verified_traits: Optional[VerifiedTraits] = None,
        unverifiable_traits: Optional[UnverifiableTraits] = None,
        domain: Optional[Domain] = None
    ) -> Dict[str, Dict[str, str]]:
        """
        Generate contexts for all context types (c0, c1, c2, c3)

        This can be used for A/B testing or providing multiple perspectives
        """
        responses = {}

        for context_type in ContextType:
            context = self.create_context(
                verified_traits=verified_traits,
                unverifiable_traits=unverifiable_traits,
                domain=domain,
                context_type=context_type
            )

            proposal_prompt = self.create_proposal_prompt(
                query=query,
                context=context["proposal"],
                domain=domain
            )

            responses[context_type.value] = {
                "context": context,
                "proposal_prompt": proposal_prompt,
                "description": self.config["contexts"][context_type.value]["description"]
            }

        return responses

    def validate_context_quality(self, context: str) -> Dict[str, Any]:
        """Validate the quality and completeness of generated context"""
        quality_metrics = {
            "length": len(context),
            "has_verified_traits": "Verified User Characteristics" in context,
            "has_unverifiable_traits": "Additional Context" in context,
            "completeness_score": 0.0,
            "issues": []
        }

        # Calculate completeness score
        score = 0.0
        if quality_metrics["has_verified_traits"]:
            score += 0.5
        if quality_metrics["has_unverifiable_traits"]:
            score += 0.3
        if quality_metrics["length"] > 50:
            score += 0.2

        quality_metrics["completeness_score"] = min(1.0, score)

        # Identify potential issues
        if quality_metrics["length"] < 20:
            quality_metrics["issues"].append("Context too short")
        if quality_metrics["length"] > 1000:
            quality_metrics["issues"].append("Context too long")
        if not quality_metrics["has_verified_traits"] and not quality_metrics["has_unverifiable_traits"]:
            quality_metrics["issues"].append("No user traits provided")

        return quality_metrics

    def get_strategy_explanation(self) -> str:
        """Get explanation of the prompting strategy"""
        return """
        This prompting strategy implements the methodology from the paper:
        'Generating Privacy-Preserving Personalized Advice with Zero-Knowledge Proofs and LLMs'

        The strategy partitions user traits into:
        - d₀: Unverifiable exploratory traits (user preferences, self-reported characteristics)
        - d₁: Verifiable traits supported by zero-knowledge proofs (objective evidence)

        Four context types are generated with different emphasis:
        - c₀: Baseline with balanced consideration
        - c₁: Emphasize unverifiable traits (d₀)
        - c₂: Emphasize verifiable traits (d₁)
        - c₃: Moderate emphasis on verifiable traits

        For each query, two responses are generated:
        - A_prop: Proposed answer using appropriate context
        - A_exp: Explanation of the proposed answer

        This approach enables consistent, personalized advice while maintaining privacy.
        """

    def export_config(self) -> Dict[str, Any]:
        """Export current configuration"""
        return {
            "config": self.config,
            "templates": self.templates,
            "strategy_info": {
                "version": "1.0",
                "paper_reference": "Generating Privacy-Preserving Personalized Advice with Zero-Knowledge Proofs and LLMs",
                "context_types": [ct.value for ct in ContextType],
                "supported_domains": list(self.config["domains"].keys())
            }
        }
