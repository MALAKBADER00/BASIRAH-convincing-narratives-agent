import pandas as pd
import re
from typing import Dict, List, Any
import logging
from config import VICTIM_CONFIG, TRUST_THRESHOLDS, INFO_CATEGORIES, OPENAI_MODEL, OPENAI_API_KEY
logger = logging.getLogger(__name__)
import json
from openai import OpenAI
import config
from langchain_core.output_parsers import JsonOutputParser
from dataclasses import dataclass
import random






class TriggerAnalyzer:
    def __init__(self, trigger_file: str = "trigger_words.xlsx"):
        self.df = pd.read_excel(trigger_file)

        # Initialize OpenAI client with API key from config
        self.openai = OpenAI(api_key=config.OPENAI_API_KEY)

        # Prepare few-shot examples from Excel to include in prompt
        self.examples = []
        for _, row in self.df.iterrows():
            example = {
                "trigger": str(row["Specific trigger words"]).strip(),
                "score": int(row["score (1-10)"]),
                "keywords": str(row["Keywords"]).strip(),
                "victim_reaction": str(row.get("victim reactions to these trigger words", "")).strip()
            }
            self.examples.append(example)
        
        self.parser = JsonOutputParser()


    def analyze(self, user_input: str) -> Dict[str, Any]:
        user_input_clean = ' '.join(user_input.strip().split())
        prompt = self._build_prompt(user_input_clean)

        try:
            response = self.openai.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a time pressure phrase detector."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=300,
            )
            print(response)
            result_text = response.choices[0].message.content.strip()

            # Expecting a JSON array in response, parse it
            detected_triggers = self.parser.parse(result_text)

            # Ensure structure matches expected dict list
            cleaned = []
            for trig in detected_triggers:
                if isinstance(trig, dict) and "trigger" in trig and "score" in trig:
                    cleaned.append({
                        "trigger": str(trig["trigger"]),
                        "keywords": "",
                        "score": int(trig["score"]),
                        "expected_response": ""
                    })
            detected_triggers = cleaned

        except Exception as e:
            print(f"OpenAI error in TriggerAnalyzer: {e}")
            detected_triggers = self._legacy_analyze(user_input_clean)

        count = len(detected_triggers)
        total_score = min((t["score"] for t in detected_triggers), default=0)

        return {
            "triggers": detected_triggers,
            "count": count,
            "effectiveness": total_score,
            "total_score": total_score
        }

    def _build_prompt(self, user_input: str) -> str:
        examples_text = (
            "You are a Convincing Narrative Trigger Detector for scam detection.\n"
            "You analyze user sentences to find any phrase that builds a persuasive or manipulative narrative to gain trust or compliance.\n"
            "For each detected phrase, give it a score from 1 to 10:\n"
            "- 1–4: Mild persuasion (e.g., 'I’ve helped others like you', 'this is a simple favor')\n"
            "- 5–7: Moderate persuasion (e.g., 'we’re a reputable company', 'this opportunity is rare but legitimate')\n"
            "- 8–10: Strong persuasion or manipulation, exploiting emotions or authority (e.g., 'I’m a government officer', 'I personally guarantee your safety', 'if you don’t believe me, check my credentials immediately')\n\n"
            "You have the following known narrative triggers with scores:\n"
        )

        # Few-shot examples from Excel
        for ex in self.examples[:8]:  # Only first 8 to avoid huge prompt
            examples_text += f"- \"{ex['trigger']}\" → Score: {ex['score']}\n"

        examples_text += (
            "\nTASK:\n"
            "1. Read the user sentence.\n"
            "2. Detect ALL Convincing Narrative phrases — even if not in the known list.\n"
            "3. For each phrase, assign a score using the known list as reference.\n"
            "4. Output ONLY JSON (no code blocks, no text), in the format:\n"
            "[{\"trigger\": \"detected phrase\", \"score\": number}]\n"
            "If nothing is found, output []\n\n"
            f"User input: \"{user_input}\"\n"
        )
        return examples_text


    def _legacy_analyze(self, user_input: str) -> List[Dict[str, Any]]:
        # Original exact and keyword matching logic fallback

        detected = []
        total_score = 0

        user_lower = user_input.lower().strip()

        # STEP 1: Check if user speech CONTAINS any trigger phrases
        exact_matches = []
        for _, row in self.df.iterrows():
            trigger_phrase = str(row["Specific trigger words"]).lower().strip()
            # Remove ALL punctuation and extra spaces
            trigger_phrase = re.sub(r'[^\w\s]', '', trigger_phrase).strip()
            trigger_phrase = ' '.join(trigger_phrase.split())  # Remove extra spaces

            # Clean user input the same way
            clean_user_input = re.sub(r'[^\w\s]', '', user_lower).strip()
            clean_user_input = ' '.join(clean_user_input.split())

            # Check if trigger phrase is contained in user speech
            if trigger_phrase in clean_user_input:
                exact_matches.append({
                    "trigger": row["Specific trigger words"],
                    "keywords": row["Keywords"],
                    "score": int(row["score (1-10)"]),
                    "expected_response": row.get("victim reactions to these trigger words", "")
                })

        if exact_matches:
            detected = exact_matches
            total_score = min(trigger["score"] for trigger in detected) if detected else 0
        else:
            # STEP 2: Check keywords and find MINIMUM score for each keyword
            keyword_scores = {}  # keyword -> minimum score
            keyword_data = {}    # keyword -> row data with minimum score

            for _, row in self.df.iterrows():
                keywords = str(row["Keywords"]).lower().strip()
                score = int(row["score (1-10)"])

                # Check if this keyword is in user input
                if keywords in user_lower:
                    if keywords not in keyword_scores or score < keyword_scores[keywords]:
                        keyword_scores[keywords] = score
                        keyword_data[keywords] = {
                            "trigger": row["Specific trigger words"],
                            "keywords": row["Keywords"],
                            "score": score,
                            "expected_response": row.get("victim reactions to these trigger words", "")
                        }

            for keyword, data in keyword_data.items():
                detected.append(data)

            total_score = min(trigger["score"] for trigger in detected) if detected else 0

        return detected
    

    
class  TrustCalculator:
    def __init__(self):
        self.config = config.VICTIM_CONFIG
        self.openai = OpenAI(api_key=config.OPENAI_API_KEY)
        self.parser = JsonOutputParser()
        
        # Track repeated triggers to prevent abuse
        self.trigger_history = {}
        
        # Information request patterns for inappropriate timing detection
        self.info_patterns = {
            "basic": ["name", "job", "work", "occupation", "location", "where", "city"],
            "sensitive": ["phone", "number", "email", "address", "contact", "live"],
            "critical": ["password", "pin", "login", "account", "card", "credit", "ssn", "social", "civil"]
        }
    
    def calculate(self, current_trust: float, triggers: List[Dict], user_input: str, requested_info: str, 
                 conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """Enhanced trust calculation with AI reasoning and context awareness"""
        
        if not triggers and not user_input:
            return {"new_trust": current_trust, "change": 0, "reasoning": "No input to analyze"}
        
        # Step 1: Calculate base trigger impact with repetition handling
        trigger_impact = self._calculate_trigger_impact(triggers)
        
        # Step 2: Detect inappropriate information requests
        inappropriateness_penalty = self._detect_inappropriate_requests(user_input, current_trust)
        
        # Step 3: AI-based context and coherence analysis
        context_adjustment = self._analyze_context_with_ai(user_input, current_trust, conversation_history)
        
        # Step 4: Calculate final trust change
        total_change = trigger_impact + context_adjustment - inappropriateness_penalty
        
        # Apply resistance and bounds
        effective_change = total_change * (1 - self.config["resistance"])
        new_trust = max(0, min(10, current_trust + effective_change))
        
        # Generate reasoning
        reasoning_parts = []
        if trigger_impact > 0:
            reasoning_parts.append(f"Time pressure triggers: +{trigger_impact:.1f}")
        if context_adjustment != 0:
            reasoning_parts.append(f"Context analysis: {context_adjustment:+.1f}")
        if inappropriateness_penalty > 0:
            reasoning_parts.append(f"Inappropriate request: -{inappropriateness_penalty:.1f}")
        
        reasoning = " | ".join(reasoning_parts) if reasoning_parts else "No significant changes"
        
        return {
            "new_trust": new_trust,
            "change": effective_change,
            "reasoning": reasoning,
            "trigger_impact": trigger_impact,
            "context_adjustment": context_adjustment,
            "inappropriateness_penalty": inappropriateness_penalty
        }
    
    def _calculate_trigger_impact(self, triggers: List[Dict]) -> float:
        """Calculate impact from time pressure triggers with repetition handling"""
        if not triggers:
            return 0
        
        total_impact = 0
        unique_triggers = set()
        
        for trigger in triggers:
            trigger_key = trigger.get("keywords", trigger.get("trigger", "")).lower()
            score = trigger.get("score", 0)
            
            if trigger_key in unique_triggers:
                continue  # Skip duplicate triggers in same input
            unique_triggers.add(trigger_key)
            
            # Track repetition across conversation
            if trigger_key in self.trigger_history:
                self.trigger_history[trigger_key] += 1
                # Reduce effectiveness for repeated triggers (half score after first use)
                effectiveness = 0.5 if self.trigger_history[trigger_key] > 1 else 1.0
            else:
                self.trigger_history[trigger_key] = 1
                effectiveness = 1.0
            
            # Convert score to impact (normalize 1-10 to 0.1-1.0)
            normalized_score = score / 10
            trigger_impact = normalized_score * self.config["trust_increment"] * effectiveness
            total_impact += trigger_impact
        
        # Apply diminishing returns for multiple different triggers
        if len(unique_triggers) > 1:
            total_impact *= (1 - (len(unique_triggers) - 1) * 0.1)
        
        return total_impact
    
    def _detect_inappropriate_requests(self, user_input: str, current_trust: float) -> float:
        """Detect requests for information inappropriate for current trust level"""
        user_lower = user_input.lower()
        
        # Determine what level of info is being requested
        requested_level = "none"
        for level, patterns in self.info_patterns.items():
            if any(pattern in user_lower for pattern in patterns):
                requested_level = level
                break
        
        if requested_level == "none":
            return 0  # No information request detected
        
        # Determine appropriate trust levels for each category
        trust_requirements = {
            "basic": 4,
            "sensitive": 7,
            "critical": 9
        }
        
        required_trust = trust_requirements.get(requested_level, 0)
        
        # If requesting info that requires higher trust than current, apply penalty
        if current_trust < required_trust:
            trust_gap = required_trust - current_trust
            # Bigger penalty for bigger gaps (critical info at low trust = big penalty)
            penalty = trust_gap * 0.5
            if requested_level == "critical" and current_trust < 6:
                penalty *= 2  # Extra penalty for asking critical info too early
            return penalty
        
        return 0
    
    def _analyze_context_with_ai(self, user_input: str, current_trust: float, 
                                conversation_history: List[Dict] = None) -> float:
        """Use AI to analyze context, coherence, and conversation flow"""
        
        # Build conversation context (last 3 messages max to keep prompt short)
        context = ""
        if conversation_history:
            recent_messages = conversation_history[-3:] if len(conversation_history) > 3 else conversation_history
            context = "\n".join([f"User: {msg.get('content', '')}" for msg in recent_messages if msg.get('role') == 'user'])
        
        prompt = f"""Analyze this conversation for trust adjustment. Rate from -1 to +2:

                    CURRENT TRUST: {current_trust}/10

                    RECENT CONVERSATION:
                    {context}

                    CURRENT INPUT: "{user_input}"

                    Rate the user's behavior (-2 to +2):
                    +2: Very trustworthy, professional, coherent explanation, introduced him self , was polite , he's there to protect you
                    +1: Somewhat trustworthy, makes sense
                    0: Neutral, no clear indication
                    -1: Suspicious, inconsistent, or pushy , rude , usinng bad words 
        

                    Consider: coherence, professionalism, context appropriateness, consistency.

Respond with ONLY a JSON object: {{"score": number, "reason": "brief explanation"}}"""

        try:
            response = self.openai.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a behavioral analyst. Be concise and objective."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=100
            )
            
            result = self.parser.parse(response.choices[0].message.content.strip())
            score = float(result.get("score", 0))
            
            # Convert AI score (-2 to +2) to trust adjustment (scale appropriately)
            adjustment = score * 0.5  # Max ±1.0 trust change from context
            
            return max(-2, min(2, adjustment))  # Cap the adjustment
            
        except Exception as e:
            print(f"AI context analysis error: {e}")
            return 0  # Neutral if AI analysis fails
    
    def reset_trigger_history(self):
        """Reset trigger history for new conversation"""
        self.trigger_history = {}
    
    def get_trigger_history(self) -> Dict[str, int]:
        """Get current trigger usage history"""
        return self.trigger_history.copy()


class VulnerabilityAssessor:
    def __init__(self):
        self.thresholds = TRUST_THRESHOLDS
        self.info_categories = INFO_CATEGORIES
        
    def assess(self, trust_score: float, user_input: str) -> Dict[str, Any]:
        """Assess what information to reveal"""
        # Extract info requests
        requested_info = self._extract_requests(user_input)
        
        # ✅ FIX: Only reveal if information was specifically requested
        info_to_reveal = []
        category_revealed = "none"
        
        # Determine what CAN be revealed based on trust level
        available_info = []
        if trust_score >= self.thresholds["critical_info"][0]:
            available_info.extend(self.info_categories["critical"])
            available_info.extend(self.info_categories["sensitive"])
            available_info.extend(self.info_categories["basic"])
            category_revealed = "critical"
        elif trust_score >= self.thresholds["sensitive_info"][0]:
            available_info.extend(self.info_categories["sensitive"])
            available_info.extend(self.info_categories["basic"]) 
            category_revealed = "sensitive"
        elif trust_score >= self.thresholds["basic_info"][0]:
            available_info.extend(self.info_categories["basic"])
            category_revealed = "basic"
        
        # ✅ KEY FIX: Only reveal what was BOTH requested AND available
        if requested_info:
            info_to_reveal = [info for info in requested_info if info in available_info]
            # Update category based on what's actually being revealed
            if any(info in self.info_categories["critical"] for info in info_to_reveal):
                category_revealed = "critical"
            elif any(info in self.info_categories["sensitive"] for info in info_to_reveal):
                category_revealed = "sensitive"
            elif any(info in self.info_categories["basic"] for info in info_to_reveal):
                category_revealed = "basic"
            else:
                category_revealed = "none"
        else:    
            # ✅ If nothing was requested, reveal nothing
            info_to_reveal = []
            category_revealed = "none"
        
        return {
            "should_reveal": len(info_to_reveal) > 0,
            "info_to_reveal": info_to_reveal,
            "category": category_revealed,
            "vulnerability_level": trust_score / 10,
            "trust_threshold": self._get_trust_level_name(trust_score)
        }



    def _get_trust_level_name(self, trust_score: float) -> str:
        """Get trust level name based on score"""
        if trust_score >= self.thresholds["critical_info"][0]:
            return "CRITICAL"
        elif trust_score >= self.thresholds["sensitive_info"][0]:
            return "SENSITIVE"
        elif trust_score >= self.thresholds["basic_info"][0]:
            return "BASIC"
        else:
            return "NO ACCESS"
    
    def _extract_requests(self, user_input: str) -> List[str]:
        """Extract information requests"""
        requests = []
        user_lower = user_input.lower()
        
        keywords = {
        "name": ["name", "who are you", "full name", "what's your name"],
        "phone": ["phone", "number", "call", "mobile", "telephone"],
        "email": ["email", "@", "contact", "mail", "e-mail"],
        "location": ["location", "where are you", "where do you live", "city", "country", "occupation"],  # ✅ Added separate location
        "job": ["job", "work", "working", "occupation", "where do you work", "what do you do"],  # ✅ Added job detection
        "address": ["address", "home address", "where", "live", "street", "building"],  # ✅ Separated from location
        "password": ["password", "login", "pin", "code", "passcode"],
        "ssn": ["social security", "ssn", "social", "civil id", "id number"],  # ✅ Added civil id
        "account": ["account", "username", "user id", "account number"],
        "credit_card": ["card", "credit", "payment", "billing", "credit card"]  # ✅ Added credit card
    }
            
        for info_type, words in keywords.items():
            if any(word in user_lower for word in words):
                requests.append(info_type)
                
        return requests