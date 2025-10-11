class GeminiProvider:
    """Google Gemini API provider implementation."""

    def __init__(self, api_key: str):
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self.client = genai

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        options: Dict[str, Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Send a chat request to Google Gemini API."""
        # Map options to Gemini parameters
        generation_config = {}
        if options:
            if "temperature" in options:
                generation_config["temperature"] = options["temperature"]
            if "top_p" in options:
                generation_config["top_p"] = options["top_p"]

        # Create a Gemini model
        gemini_model = self.client.GenerativeModel(
            model_name=model, generation_config=generation_config
        )

        # Convert messages to Gemini format
        gemini_messages = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            gemini_messages.append({"role": role, "parts": [msg["content"]]})

        # Send the chat request
        try:
            response = gemini_model.generate_content(gemini_messages)
            return {"message": {"role": "assistant", "content": response.text}}
        except Exception as e:
            # Handle quota/exhausted API
            error_msg = str(e)
            if "quota" in error_msg.lower() or "limit" in error_msg.lower():
                return {
                    "message": {
                        "role": "assistant",
                        "content": "⚠️ Free-tier API limit reached. Please try later or upgrade your plan."
                    }
                }
            else:
                # For other errors, re-raise or handle differently
                raise