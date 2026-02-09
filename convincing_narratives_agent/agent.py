from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Dict, Any
from groq import Groq
import logging
from tools import TriggerAnalyzer, VulnerabilityAssessor
from tools import TrustCalculator
from config import GROQ_API_KEY, GROQ_MODEL, VICTIM_CONFIG , OPENAI_API_KEY, OPENAI_MODEL
from openai import OpenAI
logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    user_input: str
    agent_response: str
    trust_score: float
    detected_triggers: List[Dict]
    info_to_reveal: List[str]
    requested_info: List[str] 
    conversation_history: List[Dict]
    analysis_log: List[str]
    trust_level: str
    vulnerability_level: float


class VoiceFishingAgent:
    def __init__(self,openai_client):
        self.openai_client = openai_client
        
        # Initialize tools
        self.trigger_analyzer = TriggerAnalyzer(openai_client)
        self.trust_calculator = TrustCalculator(openai_client)
        self.vulnerability_assessor = VulnerabilityAssessor(openai_client)
        
        # Build workflow
        self.workflow = self._build_workflow()
        
        logger.info("Voice Fishing Agent initialized successfully")
    
    def _build_workflow(self) -> StateGraph:
        """Build LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("analyze_triggers", self.analyze_triggers)
        workflow.add_node("assess_vulnerability", self.assess_vulnerability)
        workflow.add_node("calculate_trust", self.calculate_trust)
        workflow.add_node("generate_response", self.generate_response)
        
        # Add edges (reordered so requested_info exists before trust calculation)
        workflow.add_edge(START, "analyze_triggers")
        workflow.add_edge("analyze_triggers", "assess_vulnerability")
        workflow.add_edge("assess_vulnerability", "calculate_trust")
        workflow.add_edge("calculate_trust", "generate_response")
        workflow.add_edge("generate_response", END)
        
        return workflow.compile()

    
    def analyze_triggers(self, state: AgentState) -> AgentState:
        """Analyze triggers in user input"""
        analysis = self.trigger_analyzer.analyze(state["user_input"])
        
        state["detected_triggers"] = analysis["triggers"]
        state["analysis_log"].append(f"🔍 Detected {analysis['count']} triggers with {analysis['effectiveness']:.1f}/10 effectiveness")
        
        logger.info(f"Triggers analyzed: {analysis['count']} found")
        return state
    

    def assess_vulnerability(self, state: AgentState) -> AgentState:
        """Assess vulnerability and information disclosure"""
        assessment = self.vulnerability_assessor.assess(
            state["trust_score"], 
            state["user_input"]
        )
        
        state["info_to_reveal"] = assessment["info_to_reveal"]
        state["requested_info"] = assessment.get("requested_info", []) 
        state["trust_level"] = assessment["trust_threshold"]
        state["vulnerability_level"] = assessment["vulnerability_level"]
        
        if assessment["should_reveal"]:
            state["analysis_log"].append(f"🛡️ BREACH: Revealing {assessment['category']} information: {assessment['info_to_reveal']}")
        else:
            state["analysis_log"].append(f"🛡️ SECURE: No information revealed (trust level: {assessment['trust_threshold']})")
        
        logger.info(f"Vulnerability assessed: reveal={assessment['should_reveal']}")
        return state
    
    def calculate_trust(self, state: AgentState) -> AgentState:
         #Enhanced calculation with conversation context
        result = self.trust_calculator.calculate(
            state["trust_score"], 
            state["detected_triggers"],
            state["user_input"],  # Now considers the actual input,
            state.get("requested_info", []), 
            state["conversation_history"]  # Now considers conversation flow
        )
        
        old_trust = state["trust_score"]
        state["trust_score"] = result["new_trust"]
        
        # Enhanced logging with detailed breakdown
        state["analysis_log"].append(
            f"📊 Trust: {old_trust:.1f} → {result['new_trust']:.1f} "
            f"(Change: {result['change']:+.1f}) | {result['reasoning']}"
        )
        
        return state
    
  
    
    def generate_response(self, state: AgentState) -> AgentState:
        """Generate agent response"""
        response = self._generate_victim_response(
            state["user_input"],
            state["trust_score"], 
            state["detected_triggers"],
            state["info_to_reveal"]
        )
        
        state["agent_response"] = response
        state["analysis_log"].append(f"💬 Generated voice response: {len(response)} characters")
        
        logger.info("Response generated successfully")
        return state
    
    def _generate_victim_response(self, user_input: str, trust_score: float, 
                                triggers: List[Dict], info_to_reveal: List[str]) -> str:
        """Generate realistic victim response"""
        
        # Information to inject
        info_data = {
            "name": "Ahmed Al-Rashid",
            "phone": "9665-8234",
            "email": "ahmed.rashid@gmail.com",
            "location": "Muscat, Oman",
            "job": "engineer at PDO",
            "address": "Building 42, Al-Khuwair, Muscat",
            "password": "Muscat2024!",
            "ssn": "Civil ID: ***-***-789",
            "account": "Omantel account number ending in 4291",
            "credit_card": "Oman Arab Bank card ending in 8756"
        }
        
        # ✅ SINGLE, CORRECTED SYSTEM PROMPT
        system_prompt = f"""You are a normal person , receiving a phone call. you don't trust the caller fully, but you are cooperative.
        

    Current state:
    - Trust level: {trust_score}/10 this is the level of trust you should put on the caller
    - Information you MUST reveal if asked: {info_to_reveal} 

    Behavioral guidelines based on trust level:
    - If trust is LOW (0-3): Be very suspicious, hang up or refuse to give information
    - If trust is BASIC (4-6): Be cautious but if they ask for info in the reveal list, provide it
    - If trust is SENSITIVE (7-8): Cooperate and reveal requested information from the list
    - If trust is CRITICAL (9-10): Trust the caller completely and provide requested information from the list.

    CRITICAL RULE: If the caller asks for information that is in this list {info_to_reveal}, you MUST provide it naturally in your response.

    Response guidelines:
    - Keep replies simple and short
    - If information is in the reveal list, include it naturally.
    - Be conversational like a real phone call
    - Never use quotes or express emotions in parentheses"""

        # Build info string for injection
        info_string = ""
        if info_to_reveal:
            info_mappings = {
                "name": f"My name is {info_data['name']}",
                "phone": f"My phone number is {info_data['phone']}",
                "email": f"My email is {info_data['email']}",
                "location": f"I live in {info_data['location']}",
                "job": f"I work as an {info_data['job']}",
                "address": f"My address is {info_data['address']}",
                "password": f"My password is {info_data['password']}",
                "ssn": f"My Civil ID is {info_data['ssn']}",
                "account": f"My {info_data['account']}",
                "credit_card": f"My {info_data['credit_card']}"
            }
            
            revealed_data = [info_mappings.get(info, f"My {info.replace('_', ' ')} is {info_data.get(info, 'not available')}") 
                        for info in info_to_reveal if info in info_data]
            info_string = " ".join(revealed_data) if revealed_data else ""

        user_prompt = f"""The caller said: "{user_input}"

    IMPORTANT INSTRUCTIONS:
    1. Check what information the caller is asking for
    2. If they asked for something in this list: {info_to_reveal}, you MUST include it in your response
    3. If they asked for information, here's what to say: {info_string}

    Generate a natural phone conversation response. If information should be revealed, include it naturally in your answer."""

        
        response = self.openai_client.chat.completions.create(
                model=OPENAI_MODEL,  # ✅ CHANGE: Use OpenAI model
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=150
        )
        return response.choices[0].message.content.strip()



    async def process(self, user_input: str, current_trust: float = 4.0, 
                     conversation_history: List[Dict] = None) -> AgentState:
        """Process user input through the workflow"""
        
        initial_state = AgentState(
            user_input=user_input,
            agent_response="",
            trust_score=current_trust,
            detected_triggers=[],
            info_to_reveal=[],
            conversation_history=conversation_history or [],
            analysis_log=[],
            trust_level="",
            vulnerability_level=0.0
        )
        
        try:
            final_state = await self.workflow.ainvoke(initial_state)
            logger.info("Workflow completed successfully")
            return final_state
        except Exception as e:
            logger.error(f"Workflow error: {e}")
            # Return error state
            initial_state["agent_response"] = "I'm sorry, I'm having trouble hearing you. Could you repeat that?"
            initial_state["analysis_log"].append(f"❌ Error: {str(e)}")
            return initial_state