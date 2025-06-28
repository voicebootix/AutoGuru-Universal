# Universal Persona Factory

The Universal Persona Factory is an AI-powered module that generates authentic, engaging personas for ANY business niche automatically. It creates platform-optimized personas based on content analysis without any hardcoded business logic.

## Overview

The persona factory uses advanced AI (OpenAI GPT-4 or Anthropic Claude) to:
- Generate universal business personas that work for any niche
- Adapt personas for specific social media platforms
- Create specialized personas (educator, wellness, creative, business)
- Maintain consistency while optimizing for each platform

## Key Features

### 🎭 Universal Persona Generation
- Works for ANY business niche automatically
- AI determines strategies, never hardcoded
- Integrates with ContentAnalysis results
- Generates platform-specific adaptations

### 🎯 Persona Types

1. **BusinessPersona** - Base universal persona
2. **EducatorPersona** - For knowledge-sharing businesses
3. **WellnessPersona** - For health and fitness businesses
4. **CreativePersona** - For artistic and design businesses
5. **PlatformPersona** - Platform-specific adaptations

### 🔧 Persona Archetypes
- EDUCATOR - Knowledge sharing and teaching
- AUTHORITY - Expert and thought leader
- FRIEND - Relatable and approachable
- INNOVATOR - Creative and forward-thinking
- MOTIVATOR - Inspiring and encouraging
- CURATOR - Content curation and filtering
- STORYTELLER - Narrative-driven engagement
- PROBLEM_SOLVER - Solution-focused approach

### 💬 Communication Styles
- PROFESSIONAL - Formal and business-oriented
- CONVERSATIONAL - Casual and friendly
- INSPIRATIONAL - Motivating and uplifting
- ANALYTICAL - Data-driven and logical
- CREATIVE - Artistic and imaginative
- SUPPORTIVE - Empathetic and helpful
- ENTERTAINING - Fun and engaging
- EDUCATIONAL - Informative and teaching

## Usage

### Basic Persona Generation

```python
from backend.core.persona_factory import UniversalPersonaFactory
from backend.models.content_models import ContentAnalysis

# Initialize the factory
factory = UniversalPersonaFactory()

# Generate persona from content analysis
content_analysis = await analyze_content(...)  # Your content analysis
persona = await factory.generate_persona(content_analysis)

# Access persona attributes
print(f"Persona: {persona.name}")
print(f"Archetype: {persona.archetype.value}")
print(f"Style: {persona.style.value}")
print(f"Traits: {', '.join(persona.traits)}")
print(f"Business Focus: {persona.business_focus}")
```

### Platform-Specific Adaptation

```python
from backend.models.content_models import Platform

# Adapt persona for Instagram
instagram_persona = await factory.adapt_persona_for_platform(
    persona, 
    Platform.INSTAGRAM
)

# Access platform-specific attributes
print(f"Emoji Usage: {instagram_persona.platform_voice['emoji_usage']}")
print(f"Content Types: {instagram_persona.content_strategy['primary_content_types']}")
print(f"Engagement Tactics: {instagram_persona.engagement_tactics}")
```

### Generate Multi-Platform Personas

```python
# Generate personas for all platforms at once
platforms = [Platform.INSTAGRAM, Platform.LINKEDIN, Platform.TWITTER]
platform_personas = await factory.generate_multi_platform_personas(
    content_analysis,
    platforms
)

# Access each platform's persona
for platform, persona in platform_personas.items():
    print(f"\n{platform.value} Persona:")
    print(f"- Base: {persona.base_persona.name}")
    print(f"- Voice: {persona.platform_voice}")
    print(f"- Strategy: {persona.content_strategy}")
```

### Create Specialized Personas

#### Educator Persona
```python
educator = await factory.create_educator_persona(
    subject_area="Data Science",
    teaching_style="Interactive and practical"
)

print(f"Teaching Philosophy: {educator.teaching_philosophy}")
print(f"Learning Frameworks: {educator.learning_frameworks}")
```

#### Wellness Persona
```python
wellness = await factory.create_wellness_persona(
    wellness_type="Holistic health",
    approach="Gentle and sustainable"
)

print(f"Wellness Philosophy: {wellness.wellness_philosophy}")
print(f"Transformation Approach: {wellness.transformation_approach}")
```

#### Creative Persona
```python
creative = await factory.create_creative_persona(
    creative_field="Photography",
    style="Documentary and emotional"
)

print(f"Creative Philosophy: {creative.creative_philosophy}")
print(f"Artistic Influences: {creative.artistic_influences}")
```

## Example Output

### Fitness Business Persona

```json
{
    "name": "Coach Alex",
    "archetype": "motivator",
    "style": "inspirational",
    "traits": ["energetic", "supportive", "knowledgeable", "authentic", "results-driven"],
    "values": ["health", "personal growth", "community", "authenticity"],
    "voice_characteristics": {
        "tone_modifiers": ["encouraging", "upbeat"],
        "language_patterns": ["action verbs", "inclusive language"],
        "signature_phrases": ["Let's crush it!", "You've got this!"],
        "conversation_starters": ["How's your energy today?", "Ready for a challenge?"]
    },
    "content_preferences": {
        "favorite_topics": ["workout tips", "nutrition hacks", "mindset"],
        "content_formats": ["video", "carousel", "stories"],
        "storytelling_style": "Personal transformation narratives",
        "humor_level": "light"
    },
    "business_focus": "Holistic fitness transformation",
    "expertise_areas": ["strength training", "nutrition", "mindset coaching"],
    "unique_perspective": "Fitness is a journey of self-discovery, not just physical change",
    "engagement_approach": "Building a supportive community through shared challenges"
}
```

### Instagram Platform Adaptation

```json
{
    "platform_voice": {
        "tone_adjustment": "More visual and inspirational",
        "language_style": "Short, punchy captions with emojis",
        "emoji_usage": "moderate",
        "hashtag_strategy": "Mix of branded and trending fitness hashtags",
        "mention_strategy": "Tag clients in transformation posts"
    },
    "content_strategy": {
        "primary_content_types": ["reels", "carousel posts"],
        "posting_frequency": "1-2 times daily",
        "optimal_length": "15-30 second reels, 5-7 slide carousels",
        "visual_style": "Bright, energetic, before/after focused",
        "engagement_hooks": ["Questions in captions", "Polls in stories", "Challenge invitations"]
    },
    "engagement_tactics": [
        "Reply to all comments within 2 hours",
        "Use stories for behind-the-scenes content",
        "Host weekly Q&A sessions",
        "Create shareable quote graphics"
    ],
    "platform_specific_traits": [
        "Visual storyteller",
        "Hashtag strategist",
        "Community builder"
    ]
}
```

## Integration with Content Analysis

The persona factory seamlessly integrates with the content analysis results:

```python
# Content analysis provides:
# - Business niche detection
# - Target audience profile
# - Brand voice analysis
# - Content themes

# Persona factory uses this to create:
# - Personas that authentically represent the business
# - Personas that resonate with the target audience
# - Consistent voice across all content
# - Platform-optimized adaptations
```

## Best Practices

1. **Always Use Content Analysis**: Generate personas based on actual content analysis for authenticity
2. **Platform Adaptation**: Always adapt personas for specific platforms rather than using generic ones
3. **Regular Updates**: Regenerate personas periodically as your business evolves
4. **Consistency**: Use the same base persona across all platforms for brand consistency
5. **Test and Iterate**: Monitor engagement and adjust persona parameters as needed

## Error Handling

The factory includes comprehensive error handling:

```python
try:
    persona = await factory.generate_persona(content_analysis)
except ValueError as e:
    # Handle missing or invalid data
    logger.error(f"Invalid input: {e}")
except Exception as e:
    # Handle API or other errors
    logger.error(f"Persona generation failed: {e}")
```

## Configuration

The factory uses settings from `backend/config/settings.py`:

```python
# Required settings:
AI_OPENAI_API_KEY="your-openai-key"
AI_ANTHROPIC_API_KEY="your-anthropic-key"
AI_OPENAI_MODEL="gpt-4-turbo-preview"
AI_ANTHROPIC_MODEL="claude-3-opus-20240229"
```

## Testing

Run the comprehensive test suite:

```bash
pytest tests/unit/test_persona_factory.py -v
```

## Future Enhancements

- Voice cloning for audio content
- Video persona animations
- Multi-language persona adaptations
- Real-time persona adjustments based on engagement
- A/B testing different persona variations

---

Remember: This module is designed to work universally for ANY business niche. Always ask yourself: "Does this work for a fitness coach AND a business consultant AND an artist?"