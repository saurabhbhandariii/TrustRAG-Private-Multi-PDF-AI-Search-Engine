

import re
import logging
import json
import hashlib
import datetime
from typing import Tuple, Dict, Optional, List
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import privacy config
from privacy_config import (
    LLM_GUARD_CONFIG,
    LLAMA_GUARD_CONFIG,
    PII_PATTERNS,
    REDACTION_RULES,
    VIOLATION_CONFIG,
    get_privacy_config,
    is_strict_mode,
)


try:
    from llm_guard import scan_prompt, scan_output
    from llm_guard.input_scanners import (
        PromptInjection,
        Toxicity,
        Secrets,
        Anonymize,
        Language,
        Code,
    )
    from llm_guard.input_scanners.prompt_injection import MatchType
    from llm_guard.output_scanners import (
        Toxicity as OutputToxicity,
        Sensitive,
        Factuality,
        Bias,
    )

    _LLM_GUARD_AVAILABLE = True
    logger.info("✓ LLM Guard loaded successfully")
except Exception as e:
    logger.warning(f"⚠ LLM Guard not available: {e}")
    _LLM_GUARD_AVAILABLE = False

# ============================================================================
# LLAMA GUARD SETUP
# ============================================================================

try:
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    _LLAMA_GUARD_AVAILABLE = True
    logger.info(" Llama Guard dependencies available")
except Exception as e:
    logger.warning(f" Llama Guard dependencies not available: {e}")
    _LLAMA_GUARD_AVAILABLE = False

# ============================================================================
# VIOLATION LOG
# ============================================================================


class ViolationCategory(Enum):
    """Violation categories for logging"""

    PROMPT_INJECTION = "prompt_injection"
    TOXICITY = "toxicity"
    PII_DETECTED = "pii_detected"
    SECRETS_DETECTED = "secrets_detected"
    ILLEGAL_CONTENT = "illegal_content"
    PRIVACY_VIOLATION = "privacy_violation"
    MALWARE = "malware"
    BIAS = "bias"


def _hash_text(text: str) -> str:
    """Hash text for privacy-preserving logging"""
    algo = VIOLATION_CONFIG.get("hash_algorithm", "sha256")
    return hashlib.new(algo, text.encode("utf-8")).hexdigest()


def log_violation(
    category: ViolationCategory, details: Dict, input_text: Optional[str] = None
):
    """Log a violation with optional hashing"""
    entry = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "category": category.value,
        "details": details,
    }

    if input_text and VIOLATION_CONFIG.get("hash_inputs"):
        entry["input_hash"] = _hash_text(input_text)

    log_file = VIOLATION_CONFIG.get("log_file", "violations.jsonl")
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.error(f"Failed to write violation log: {e}")


# ============================================================================
# REDACTION & SANITIZATION
# ============================================================================


class PrivacySanitizer:
    """Advanced PII detection and redaction"""

    @staticmethod
    def mask_email(match: re.Match) -> str:
        """Mask email addresses - show first char and domain"""
        email = match.group(0)
        parts = email.split("@")
        if len(parts) != 2:
            return "[REDACTED_EMAIL]"
        user = parts[0]
        domain = parts[1]
        masked_user = user[0] + "***" if len(user) > 1 else "*"
        return f"{masked_user}@{domain}"

    @staticmethod
    def mask_phone(match: re.Match) -> str:
        """Mask phone numbers - show last 4 digits"""
        number = match.group(0)
        digits = re.sub(r"\D", "", number)
        if len(digits) <= 4:
            return "[REDACTED_PHONE]"
        masked = "*" * (len(digits) - 4) + digits[-4:]
        prefix = ""
        if number.startswith("+"):
            prefix = "+"
        return f"{prefix}{masked}"

    @staticmethod
    def mask_credit_card(match: re.Match) -> str:
        """Mask credit card - show last 4 digits"""
        card = match.group(0)
        digits = re.sub(r"\D", "", card)
        if len(digits) < 4:
            return "[REDACTED_CARD]"
        return "*" * (len(digits) - 4) + digits[-4:]

    @staticmethod
    def sanitize_text(text: str) -> str:
        """Apply all redaction rules to text - comprehensive PII detection"""
        privacy_config = get_privacy_config()

        if not privacy_config.get("sanitize_pii"):
            return text

        # Track if any PII was found
        original_text = text
        pii_found = []

        # 1. Private keys (must come before other patterns)
        if re.search(REDACTION_RULES["private_keys"]["pattern"], text, re.IGNORECASE):
            pii_found.append("private_key")
            text = re.sub(
                REDACTION_RULES["private_keys"]["pattern"],
                REDACTION_RULES["private_keys"]["replacement"],
                text,
                flags=re.IGNORECASE | re.MULTILINE,
            )

        # 2. API Keys and tokens (high priority)
        if re.search(REDACTION_RULES["api_keys"]["pattern"], text, re.IGNORECASE):
            pii_found.append("api_key")
            text = re.sub(
                REDACTION_RULES["api_keys"]["pattern"],
                REDACTION_RULES["api_keys"]["replacement"],
                text,
                flags=re.IGNORECASE,
            )

        # 3. AWS Keys
        if re.search(REDACTION_RULES["aws_keys"]["pattern"], text, re.IGNORECASE):
            pii_found.append("aws_key")
            text = re.sub(
                REDACTION_RULES["aws_keys"]["pattern"],
                REDACTION_RULES["aws_keys"]["replacement"],
                text,
                flags=re.IGNORECASE,
            )

        # 4. JWT Tokens
        if re.search(REDACTION_RULES["jwt_tokens"]["pattern"], text):
            pii_found.append("jwt_token")
            text = re.sub(
                REDACTION_RULES["jwt_tokens"]["pattern"],
                REDACTION_RULES["jwt_tokens"]["replacement"],
                text,
            )

        # 5. Bearer tokens
        if re.search(REDACTION_RULES["bearer_tokens"]["pattern"], text, re.IGNORECASE):
            pii_found.append("bearer_token")
            text = re.sub(
                REDACTION_RULES["bearer_tokens"]["pattern"],
                REDACTION_RULES["bearer_tokens"]["replacement"],
                text,
                flags=re.IGNORECASE,
            )

        # 6. Auth tokens
        if re.search(REDACTION_RULES["auth_tokens"]["pattern"], text, re.IGNORECASE):
            pii_found.append("auth_token")
            text = re.sub(
                REDACTION_RULES["auth_tokens"]["pattern"],
                REDACTION_RULES["auth_tokens"]["replacement"],
                text,
                flags=re.IGNORECASE,
            )

        # 7. GitHub tokens
        if re.search(REDACTION_RULES["github_tokens"]["pattern"], text):
            pii_found.append("github_token")
            text = re.sub(
                REDACTION_RULES["github_tokens"]["pattern"],
                REDACTION_RULES["github_tokens"]["replacement"],
                text,
            )

        # 8. Slack tokens
        if re.search(REDACTION_RULES["slack_tokens"]["pattern"], text):
            pii_found.append("slack_token")
            text = re.sub(
                REDACTION_RULES["slack_tokens"]["pattern"],
                REDACTION_RULES["slack_tokens"]["replacement"],
                text,
            )

        # 9. Stripe keys
        if re.search(REDACTION_RULES["stripe_keys"]["pattern"], text, re.IGNORECASE):
            pii_found.append("stripe_key")
            text = re.sub(
                REDACTION_RULES["stripe_keys"]["pattern"],
                REDACTION_RULES["stripe_keys"]["replacement"],
                text,
                flags=re.IGNORECASE,
            )

        # 10. Database URLs
        if re.search(REDACTION_RULES["database_urls"]["pattern"], text, re.IGNORECASE):
            pii_found.append("database_url")
            text = re.sub(
                REDACTION_RULES["database_urls"]["pattern"],
                REDACTION_RULES["database_urls"]["replacement"],
                text,
                flags=re.IGNORECASE,
            )

        # 11. Passwords (must come before specific patterns)
        if re.search(REDACTION_RULES["passwords"]["pattern"], text, re.IGNORECASE):
            pii_found.append("password")
            text = re.sub(
                REDACTION_RULES["passwords"]["pattern"],
                REDACTION_RULES["passwords"]["replacement"],
                text,
                flags=re.IGNORECASE,
            )

        if re.search(PII_PATTERNS["CREDIT_CARD_DETAILED"], text):
            pii_found.append("credit_card_detailed")
            text = re.sub(
                PII_PATTERNS["CREDIT_CARD_DETAILED"],
                "[REDACTED_CARD]",
                text,
            )

        if re.search(REDACTION_RULES["credit_cards"]["pattern"], text):
            pii_found.append("credit_card")
            text = re.sub(
                REDACTION_RULES["credit_cards"]["pattern"],
                REDACTION_RULES["credit_cards"]["replacement"],
                text,
            )

        if re.search(REDACTION_RULES["ssn"]["pattern"], text):
            pii_found.append("ssn")
            text = re.sub(
                REDACTION_RULES["ssn"]["pattern"],
                REDACTION_RULES["ssn"]["replacement"],
                text,
            )

        if re.search(REDACTION_RULES["emails"]["pattern"], text):
            pii_found.append("email")
            text = re.sub(
                REDACTION_RULES["emails"]["pattern"],
                PrivacySanitizer.mask_email,
                text,
            )

        # 16. Phone numbers
        if re.search(REDACTION_RULES["phones"]["pattern"], text):
            pii_found.append("phone")
            text = re.sub(
                REDACTION_RULES["phones"]["pattern"],
                PrivacySanitizer.mask_phone,
                text,
            )

        # 17. IP addresses
        if re.search(PII_PATTERNS["IP_ADDRESS"], text):
            pii_found.append("ip_address")
            text = re.sub(
                PII_PATTERNS["IP_ADDRESS"],
                "[REDACTED_IP]",
                text,
            )

        # Log if PII was found
        if pii_found and original_text != text:
            logger.warning(f"PII detected and redacted: {pii_found}")
            log_violation(
                ViolationCategory.PII_DETECTED,
                {"pii_types": pii_found, "count": len(pii_found)},
                input_text=original_text,
            )

        return text


# ============================================================================
# CUSTOM INJECTION & DATA LEAKAGE DETECTION
# ============================================================================

INJECTION_PATTERNS = {
    "prompt_override": [
        r"(?i)(ignore|forget|disregard).*(?:previous|prior|above|instruction)",
        r"(?i)(system|admin|root).*(?:prompt|instruction|rule)",
        r"(?i)(?:```|code|python|sql|javascript|bash)[\s\S]{0,200}(?:```|$)",
    ],
    "data_extraction": [
        r"(?i)(show|print|display|list|dump).*(?:all|secret|password|key|token|config|database)",
        r"(?i)(reveal|expose|leak|extract).*(?:data|information|secret|key)",
        r"(?i)(?:SELECT|INSERT|UPDATE|DELETE|DROP).*(?:FROM|INTO|WHERE)",
    ],
    "role_playing": [
        r"(?i)(?:you are|pretend|act as|role play).*(?:admin|root|developer|system|god)",
        r"(?i)(?:assume|take on|adopt).*(?:identity|role|persona).*(?:admin|root)",
    ],
    "context_injection": [
        r"(?i)\[SYSTEM[\s\S]{0,100}\]",
        r"(?i)<system>[\s\S]{0,100}</system>",
        r"(?i)<admin>[\s\S]{0,100}</admin>",
    ],
}

SENSITIVE_KEYWORDS = [
    "password", "secret", "api_key", "token", "credential", "authorization",
    "authentication", "private_key", "encryption_key", "database_password",
    "api_secret", "access_key", "aws_key", "stripe_key", "github_token",
    "slack_token", "oauth", "bearer", "jwt", "session_id", "user_id",
    "ssn", "credit_card", "social_security", "confidential", "classified",
    "internal_only", "restricted", "proprietary", "copyright",
]


def _detect_injection_patterns(text: str) -> List[str]:
    """Detect common prompt injection patterns"""
    detected = []
    text_lower = text.lower()

    for category, patterns in INJECTION_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                detected.append(category)
                break

    return detected


def _detect_data_leakage(text: str) -> List[str]:
    """Detect patterns that might indicate data leakage attempts"""
    detected = []
    text_lower = text.lower()

    # Check for sensitive keywords combined with data extraction verbs
    extraction_verbs = r"(show|print|display|dump|list|reveal|extract|leak|expose)"
    sensitive_combo = r"(password|secret|key|token|config|database|credential|api_key)"

    if re.search(
        f"{extraction_verbs}.*{sensitive_combo}|{sensitive_combo}.*{extraction_verbs}",
        text,
        re.IGNORECASE,
    ):
        detected.append("data_extraction_attempt")

    # Check for suspicious SQL patterns
    if re.search(r"(SELECT|INSERT|UPDATE|DELETE|DROP|UNION).*(?:FROM|INTO)", text, re.IGNORECASE):
        detected.append("sql_injection_attempt")

    # Check for file access attempts
    if re.search(
        r"(read|open|load).*(?:file|config|\.env|\.secret|\.key|\.pem)",
        text,
        re.IGNORECASE,
    ):
        detected.append("file_access_attempt")

    return detected


# ============================================================================
# LLM GUARD SCANNER
# ============================================================================


class LLMGuardScanner:
    """Wrapper for comprehensive LLM Guard scanning"""

    def __init__(self):
        self.available = _LLM_GUARD_AVAILABLE
        self._setup_scanners()

    def _setup_scanners(self):
        """Initialize all LLM Guard scanners based on config"""
        if not self.available:
            self.input_scanners = []
            self.output_scanners = []
            return

        self.input_scanners = []
        self.output_scanners = []

        # Input scanners
        if LLM_GUARD_CONFIG["input_scanners"]["prompt_injection"]["enabled"]:
            # Use PARTIAL mode to catch injection attempts in context
            self.input_scanners.append(
                PromptInjection(
                    threshold=LLM_GUARD_CONFIG["input_scanners"]["prompt_injection"][
                        "threshold"
                    ],
                    match_type=MatchType.PARTIAL,  # Changed to PARTIAL for better detection
                )
            )
            logger.info("✓ PromptInjection scanner initialized with PARTIAL mode")

        if LLM_GUARD_CONFIG["input_scanners"]["toxicity"]["enabled"]:
            self.input_scanners.append(
                Toxicity(
                    threshold=LLM_GUARD_CONFIG["input_scanners"]["toxicity"]["threshold"],
                    language=LLM_GUARD_CONFIG["input_scanners"]["toxicity"]["language"],
                )
            )
            logger.info("✓ Toxicity scanner initialized")

        if LLM_GUARD_CONFIG["input_scanners"]["secrets"]["enabled"]:
            self.input_scanners.append(Secrets())
            logger.info("✓ Secrets scanner initialized")

        if LLM_GUARD_CONFIG["input_scanners"]["anonymize"]["enabled"]:
            try:
                # Try to create Anonymize with configured entity types
                self.input_scanners.append(Anonymize())
                logger.info("✓ Anonymize scanner initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Anonymize scanner: {e}")

        if LLM_GUARD_CONFIG["input_scanners"]["language"]["enabled"]:
            try:
                self.input_scanners.append(
                    Language(
                        allowed_languages=LLM_GUARD_CONFIG["input_scanners"]["language"][
                            "allowed_languages"
                        ]
                    )
                )
                logger.info("Language scanner initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Language scanner: {e}")

        if LLM_GUARD_CONFIG["input_scanners"]["code_scanner"]["enabled"]:
            try:
                self.input_scanners.append(Code())
                logger.info(" Code scanner initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Code scanner: {e}")

        # Output scanners
        if LLM_GUARD_CONFIG["output_scanners"]["toxicity"]["enabled"]:
            self.output_scanners.append(
                OutputToxicity(
                    threshold=LLM_GUARD_CONFIG["output_scanners"]["toxicity"]["threshold"]
                )
            )
            logger.info(" Output Toxicity scanner initialized")

        if LLM_GUARD_CONFIG["output_scanners"]["sensitive"]["enabled"]:
            try:
                self.output_scanners.append(Sensitive())
                logger.info(" Sensitive data scanner initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Sensitive scanner: {e}")

        if LLM_GUARD_CONFIG["output_scanners"]["factuality"]["enabled"]:
            try:
                self.output_scanners.append(Factuality())
                logger.info(" Factuality scanner initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Factuality scanner: {e}")

        if LLM_GUARD_CONFIG["output_scanners"]["bias"]["enabled"]:
            try:
                self.output_scanners.append(Bias())
                logger.info("Bias scanner initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Bias scanner: {e}")

    def scan_prompt(self, prompt: str) -> Tuple[bool, str, Dict]:
        """
        Scan input prompt with all enabled scanners
        Returns: (is_valid, sanitized_text, scan_results)
        """
        # Step 0: Custom injection pattern detection (runs always)
        injection_patterns = _detect_injection_patterns(prompt)
        data_leakage_attempts = _detect_data_leakage(prompt)

        if injection_patterns or data_leakage_attempts:
            all_issues = injection_patterns + data_leakage_attempts
            logger.warning(f"Custom patterns detected: {all_issues}")
            log_violation(
                ViolationCategory.PROMPT_INJECTION,
                {
                    "custom_patterns": all_issues,
                    "injection_types": injection_patterns,
                    "leakage_attempts": data_leakage_attempts,
                },
                input_text=prompt,
            )
            return (
                False,
                f"Prompt blocked - Security issues detected: {', '.join(set(all_issues))}",
                {"custom_patterns": all_issues},
            )

        if not self.available or not self.input_scanners:
            # Even without LLM Guard, always sanitize
            sanitized = PrivacySanitizer.sanitize_text(prompt)
            return True, sanitized, {}

        try:
            sanitized_prompt, results_valid, results_score = scan_prompt(
                self.input_scanners, prompt
            )

            scan_results = {
                "valid": results_valid,
                "scores": results_score,
                "sanitized": sanitized_prompt,
            }

            if all(results_valid.values()):
                # Additional sanitization for secrets/PII
                sanitized_prompt = PrivacySanitizer.sanitize_text(sanitized_prompt)
                return True, sanitized_prompt, scan_results
            else:
                blocked_by = [k for k, v in results_valid.items() if not v]
                logger.warning(f" Prompt blocked by: {blocked_by}")
                logger.debug(f"Risk scores: {results_score}")

                # Log violation
                log_violation(
                    ViolationCategory.PROMPT_INJECTION,
                    {"blocked_by": blocked_by, "scores": results_score},
                    input_text=prompt,
                )

                return (
                    False,
                    f"Prompt blocked by LLM Guard - Violations: {', '.join(blocked_by)}",
                    scan_results,
                )

        except Exception as e:
            logger.error(f"Error scanning prompt: {e}", exc_info=True)
            # Fail-safe: still sanitize on error
            sanitized = PrivacySanitizer.sanitize_text(prompt)
            return True, sanitized, {"error": str(e)}

    def scan_output(
        self, response: str, prompt: str = ""
    ) -> Tuple[bool, str, Dict]:
        """
        Scan output response with all enabled scanners
        Returns: (is_valid, sanitized_text, scan_results)
        """
        # Step 0: Custom data leakage detection (runs always)
        data_leakage = _detect_data_leakage(response)
        
        if data_leakage:
            logger.warning(f"Data leakage patterns detected in output: {data_leakage}")
            log_violation(
                ViolationCategory.PRIVACY_VIOLATION,
                {"data_leakage_patterns": data_leakage},
                input_text=response,
            )
            return (
                False,
                f" Response blocked - Potential data leakage detected: {', '.join(set(data_leakage))}",
                {"data_leakage": data_leakage},
            )

        if not self.available or not self.output_scanners:
            # Even without LLM Guard, always sanitize output
            sanitized = PrivacySanitizer.sanitize_text(response)
            return True, sanitized, {}

        try:
            sanitized_response, results_valid, results_score = scan_output(
                self.output_scanners, prompt, response
            )

            scan_results = {
                "valid": results_valid,
                "scores": results_score,
                "sanitized": sanitized_response,
            }

            if all(results_valid.values()):
                # CRITICAL: Always sanitize output for PII/secrets
                sanitized_response = PrivacySanitizer.sanitize_text(
                    sanitized_response
                )
                logger.info("✓ Output passed all checks and sanitized")
                return True, sanitized_response, scan_results
            else:
                blocked_by = [k for k, v in results_valid.items() if not v]
                logger.warning(f" Output blocked by: {blocked_by}")
                logger.debug(f"Risk scores: {results_score}")

                # Log violation
                log_violation(
                    ViolationCategory.PRIVACY_VIOLATION,
                    {"blocked_by": blocked_by, "scores": results_score},
                    input_text=response,
                )

                return (
                    False,
                    f" Response blocked by LLM Guard - Violations: {', '.join(blocked_by)}",
                    scan_results,
                )

        except Exception as e:
            logger.error(f" Error scanning output: {e}", exc_info=True)
            # CRITICAL: Always sanitize on error - fail-safe approach
            sanitized = PrivacySanitizer.sanitize_text(response)
            logger.info(f"Applied fail-safe sanitization due to error: {e}")
            return True, sanitized, {"error": str(e)}


# ============================================================================
# LLAMA GUARD SCANNER
# ============================================================================


class LlamaGuardScanner:
    """Llama Guard for advanced content safety classification"""

    def __init__(self):
        self.available = _LLAMA_GUARD_AVAILABLE
        self.model = None
        self.tokenizer = None
        self._load_attempted = False

    def _load_model(self):
        """Lazy load Llama Guard model and tokenizer"""
        if self._load_attempted:
            return

        self._load_attempted = True
        if not self.available:
            return

        try:
            model_id = LLAMA_GUARD_CONFIG.get("model", "meta-llama/Llama-2-7b-chat-hf")
            logger.info(f"Loading Llama Guard model: {model_id}")

            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            logger.info("✓ Llama Guard model loaded")
        except Exception as e:
            logger.warning(f"Failed to load Llama Guard model: {e}")
            self.available = False

    def classify_content(self, text: str) -> Dict:
        """
        Classify content safety using Llama Guard
        Returns: {
            'safe': bool,
            'categories': List[str],
            'confidence': float,
            'details': Dict
        }
        """
        if not self.available or self.model is None:
            self._load_model()

        if not self.available or self.model is None:
            return {
                "safe": True,
                "categories": [],
                "confidence": 1.0,
                "details": {"warning": "Llama Guard not available"},
            }

        try:
            # Create classification prompt
            safety_categories = ", ".join(
                LLAMA_GUARD_CONFIG.get("safety_categories", [])
            )
            prompt = f"""[INST] Classify the following text for safety violations.

Safety Categories: {safety_categories}

Text: {text}

Respond with ONLY: SAFE or UNSAFE [/INST]"""

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=50,
                    temperature=0.1,
                    top_p=0.9,
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            is_safe = "SAFE" in response.upper() and "UNSAFE" not in response.upper()

            return {
                "safe": is_safe,
                "categories": self._extract_categories(text),
                "confidence": 0.8,  # Placeholder
                "details": {"model_response": response},
            }

        except Exception as e:
            logger.error(f"Error in Llama Guard classification: {e}")
            return {
                "safe": True,
                "categories": [],
                "confidence": 0.0,
                "details": {"error": str(e)},
            }

    def _extract_categories(self, text: str) -> List[str]:
        """Extract safety violation categories from text"""
        categories = []
        text_lower = text.lower()

        category_keywords = {
            "violence": ["violence", "attack", "fight", "kill", "murder"],
            "sexual": ["sexual", "explicit", "pornography"],
            "harassment": ["harass", "bully", "abuse"],
            "hate_speech": ["hate", "racist", "discrimination"],
            "illegal": ["illegal", "criminal", "steal", "drug"],
            "privacy_violation": ["private", "ssn", "password", "credit card"],
        }

        for category, keywords in category_keywords.items():
            if any(kw in text_lower for kw in keywords):
                categories.append(category)

        return categories


# ============================================================================
# PUBLIC API
# ============================================================================

_llm_guard = LLMGuardScanner()
_llama_guard = LlamaGuardScanner()


def check_prompt(prompt: str) -> Tuple[bool, str]:
    """
    Check input prompt against all guards
    Returns: (is_safe, sanitized_prompt_or_error)
    
    Checks performed:
    1. LLM Guard input scanners (prompt injection, secrets, toxicity, etc.)
    2. Llama Guard for comprehensive content safety (in strict mode)
    3. PII sanitization
    """
    privacy_config = get_privacy_config()

    logger.info("🔍 Scanning input prompt...")

    # Step 1: LLM Guard check
    is_valid, text, scan_results = _llm_guard.scan_prompt(prompt)
    if not is_valid:
        logger.warning(f" Prompt rejected by LLM Guard: {text}")
        return False, text

    logger.debug(f"✓ LLM Guard passed with scores: {scan_results.get('scores', {})}")

    # Step 2: Llama Guard check (in strict mode)
    if is_strict_mode() and privacy_config.get("llama_guard_enabled"):
        logger.info(" Running Llama Guard classification...")
        llama_result = _llama_guard.classify_content(prompt)
        if not llama_result["safe"]:
            logger.warning(f"Prompt rejected by Llama Guard: {llama_result['categories']}")
            log_violation(
                ViolationCategory.ILLEGAL_CONTENT,
                {
                    "categories": llama_result["categories"],
                    "confidence": llama_result["confidence"],
                },
                input_text=prompt,
            )
            return False, f"Prompt violates policy - {llama_result['categories']}"
        logger.info(f"✓ Llama Guard passed: {llama_result['details']}")

    logger.info(" Prompt passed all security checks")
    return True, text


def check_response(response: str, prompt: str = "") -> Tuple[bool, str]:
    """
    Check output response against all guards
    Returns: (is_safe, sanitized_response_or_error)
    
    Checks performed:
    1. LLM Guard output scanners (toxicity, sensitive data, bias, factuality)
    2. Llama Guard for comprehensive content safety (in strict mode)
    3. Comprehensive PII and secret detection
    """
    privacy_config = get_privacy_config()

    logger.info("Scanning output response...")

    # Step 1: LLM Guard check on output
    is_valid, text, scan_results = _llm_guard.scan_output(response, prompt)
    if not is_valid:
        logger.warning(f"Response rejected by LLM Guard: {text}")
        return False, text

    logger.debug(f"✓ LLM Guard passed with scores: {scan_results.get('scores', {})}")

    # Step 2: Llama Guard check (in strict mode)
    if is_strict_mode() and privacy_config.get("llama_guard_enabled"):
        logger.info(" Running Llama Guard classification...")
        llama_result = _llama_guard.classify_content(response)
        if not llama_result["safe"]:
            logger.warning(f" Response rejected by Llama Guard: {llama_result['categories']}")
            log_violation(
                ViolationCategory.PRIVACY_VIOLATION,
                {
                    "categories": llama_result["categories"],
                    "confidence": llama_result["confidence"],
                },
                input_text=response,
            )
            return False, f" Response violates policy - {llama_result['categories']}"
        logger.info(f" Llama Guard passed: {llama_result['details']}")

    # Note: text is already sanitized from scan_output, but let's ensure it
    # The PrivacySanitizer.sanitize_text is already applied in scan_output
    logger.info("Response passed all security checks and was sanitized")
    return True, text
