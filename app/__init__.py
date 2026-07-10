"""FourVoices — grounded multi-style video captioning with Gemma.

Captions short clips in four voices (formal, sarcastic, humorous-tech,
humorous-non-tech). Design principle: ground the visual facts ONCE with Gemma's
multimodal vision, then restyle into four tones — so tone changes but content
stays faithful. Accuracy and tone are exactly what the hackathon's LLM-judge
scores, so we optimize both explicitly with a self-check.
"""

__version__ = "0.1.0"
