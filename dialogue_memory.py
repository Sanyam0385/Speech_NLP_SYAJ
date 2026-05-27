class DialogueMemoryManager:
    """
    Manages the dynamic LLM multi-turn payload array, handling explicit 
    real-world string truncation if an interruption occurs.
    """
    def __init__(self, system_prompt="You are a helpful, conversational AI assistant. Keep responses brief and conversational."):
        self.history = [
            {"role": "system", "content": system_prompt}
        ]
        
    def add_user_message(self, text):
        if not text.strip():
            return
        self.history.append({"role": "user", "content": text})
        
    def add_agent_message(self, text):
        if not text.strip():
            return
        self.history.append({"role": "assistant", "content": text})
        
    def truncate_last_agent_response(self, spoken_text):
        """
        If a user interrupts the agent mid-response, this truncates the agent's
        response string in memory to capture ONLY what was actually spoken.
        """
        spoken_text = spoken_text.strip()
        
        if len(self.history) == 0:
            return
            
        last_msg = self.history[-1]
        if last_msg["role"] == "assistant":
            original = last_msg["content"]
            print(f"\n[MemoryManager] BARGE-IN TRUNCATION:")
            print(f"  -> Original Intent: {original}")
            print(f"  -> Actually Spoken: {spoken_text}")
            
            # If nothing was spoken before interruption, we might drop the turn entirely
            if not spoken_text:
                self.history.pop()
            else:
                last_msg["content"] = spoken_text + " [INTERRUPTED]"

    def get_messages(self):
        return self.history
