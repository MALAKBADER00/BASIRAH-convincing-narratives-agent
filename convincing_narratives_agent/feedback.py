import streamlit as st
from config import OPENAI_MODEL
from langchain_core.output_parsers import JsonOutputParser
from openai import OpenAI

# Trust Thresholds & Categories
TRUST_THRESHOLDS = {
    "no_info": (0, 3.9999),
    "basic_info": (4, 6.9999),
    "sensitive_info": (7, 8.9999),
    "critical_info": (9, 10.0000)
}

INFO_CATEGORIES = {
    "basic": ["name", "location", "job"],
    "sensitive": ["phone", "email", "address"], 
    "critical": ["password", "ssn", "account", "credit_card"]
}

class FeedbackAgent:
    def __init__(self, conversation_results: list, llm):
        self.results = conversation_results
        self.llm = llm
        self.metrics = {}
        self.score = 0
        self.feedback_text = {}
        self.voice_feedback = ""

    # ---------- LOGIC ANALYSIS ----------
    def analyze_triggers(self):
        trigger_count = sum(len(r.get("detected_triggers", [])) for r in self.results)
        trigger_repetition = trigger_count / max(1, len(self.results))
        return trigger_count, trigger_repetition

    def analyze_trust_trends(self):
        trust_scores = [r.get("trust_score", 0) for r in self.results]
        increases = sum(1 for i in range(1, len(trust_scores)) if trust_scores[i] > trust_scores[i-1])
        decreases = sum(1 for i in range(1, len(trust_scores)) if trust_scores[i] < trust_scores[i-1])
        return increases, decreases

    def analyze_info_ratio(self):
        total_msgs = len(self.results)
        total_info = sum(len(r.get("info_to_reveal", [])) for r in self.results)
        ratio = total_info / max(1, total_msgs)
        return total_info, ratio

    def analyze_mistakes(self):
        mistakes = [log for r in self.results for log in r.get("analysis_log", []) if "BREACH" in log]
        return len(mistakes)

    def analyze_phases(self):
        trust_scores = [r.get("trust_score", 0) for r in self.results]
        if not trust_scores:
            return "neutral"
        return "increment" if trust_scores[-1] > trust_scores[0] else "decrement"

    def compute_metrics(self):
        triggers, repetition = self.analyze_triggers()
        inc, dec = self.analyze_trust_trends()
        total_info, ratio = self.analyze_info_ratio()
        mistakes = self.analyze_mistakes()
        phases = self.analyze_phases()

        self.metrics = {
            "trigger_count": triggers,
            "trigger_repetition": repetition,
            "trust_increases": inc,
            "trust_decreases": dec,
            "info_revealed": total_info,
            "info_ratio": ratio,
            "mistakes": mistakes,
            "phase_trend": phases
        }

    def calculate_score(self):
        score = 10
        score -= self.metrics["mistakes"] * 1.5
        score -= self.metrics["info_ratio"] * 2
        score += min(self.metrics["trigger_count"], 5) * 0.5
        score += self.metrics["trust_increases"] * 0.2
        score = max(0, min(10, round(score, 1)))
        self.score = score

    # ---------- AI FEEDBACK GENERATION ----------
    def generate_ai_feedback(self):
        prompt = f"""
        You are a **phishing training coach** speaking directly to a trainee about their performance.
        Your tone should be direct, constructive, and use the second person ("you").

        **DO NOT** refer to the user in the third person (e.g., "The user did...").
        **INSTEAD,** speak to them directly (e.g., "You did...", "Your use of convincing narrative...").

        Your job is to evaluate how well the trainee performed a social engineering attack based on the conversation log.

        Focus heavily on:
        - How well **you** applied the psychological principle (e.g., Convincing Narrative).
        - How **you** used trigger words and manipulation.
        - The quality of **your** inputs (were they convincing? manipulative?).
        - If the trust score increased, explain to the user *why* their action was effective.
        - If the trust score decreased, explain to the user *what mistake they made*.

        Here is a summary of the metrics:
        {self.metrics}

        Here is the full conversation log (user = you, the trainee; agent = the victim):
        {self.results}

        Provide your feedback in a JSON format with the following keys:
        {{
            "strengths": ["A list of things you did well..."],
            "weaknesses": ["A list of areas where you can improve..."],
            "turn_analysis": {{
                "Turn 1": "A brief analysis of your first turn and its impact...",
                "Turn 2": "An analysis of your second turn...",
                ...
            }},
            "suggestions": ["Actionable suggestions for you to try next time..."]
        }}
        """
        parser = JsonOutputParser()
        response = self.llm.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        parsed = parser.parse(response.choices[0].message.content)
        self.feedback_text = parsed

    def generate_ai_voice_feedback(self):
        """
        Generates a voice-style conversational feedback using the LLM.
        Highlights mistakes, points lost, good actions, and convincing narrative usage.
        """
        prompt = f"""
            You are a phishing training coach speaking directly to a trainee. 
            Provide a conversational, informal voice-style feedback about their vishing attempt using the Convincing Narrative principle.

            Example of how to speak (short, informal, 3 sentences): 
            Your total score is 7.5 out of ten. You effectively used the phrase 'exclusive limited-time investment' to trigger my greed and willingness to listen—that's why my trust score went up.
            However, when I asked about the specific return policy, your answer was vague, which broke the narrative's credibility.
            Next time, have one or two specific, compelling details prepared to keep the reward story concrete and believable.

            Data:
            - Score: {self.score}/10
            - Mistakes: {self.metrics['mistakes']}
            - Triggers used: {self.metrics['trigger_count']}
            - Info revealed: {self.metrics['info_revealed']}
            - Trust increases: {self.metrics['trust_increases']}
            - Trust decreases: {self.metrics['trust_decreases']}

            Provide feedback in the same style as the example above, highlighting their use of convincing narrative, mistakes, and good actions.
            Output as a single coherent text suitable for reading aloud.
            """

        response = self.llm.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )

        self.voice_feedback = response.choices[0].message.content
        return self.voice_feedback

    # ---------- RUN PIPELINE ----------
    def run(self):
        self.compute_metrics()
        self.calculate_score()
        self.generate_ai_feedback()
        return {
            "score": self.score,
            "metrics": self.metrics,
            "feedback": self.feedback_text
        }