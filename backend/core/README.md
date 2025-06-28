# AutoGuru Universal - Core Content Analyzer

## Overview

The `UniversalContentAnalyzer` is an AI-powered content analysis system that automatically works for ANY business niche. It uses Large Language Models (LLMs) to analyze content and determine optimal social media strategies without any hardcoded business logic.

## Features

- **Universal Niche Detection**: Automatically detects business type from content (education, fitness, business consulting, creative, etc.)
- **AI-Powered Analysis**: Uses OpenAI GPT-4 or Anthropic Claude for intelligent content understanding
- **Target Audience Identification**: Determines optimal demographics and psychographics
- **Brand Voice Extraction**: Identifies communication style and tone
- **Viral Potential Assessment**: Predicts content performance across different platforms
- **Async/Await Support**: Built for high-performance, scalable applications
- **Comprehensive Error Handling**: Includes retry logic and graceful fallbacks

## Installation

```bash
# Install required dependencies
pip install -r requirements.txt

# Set up environment variables
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"  # Optional
```

## Usage

### Basic Usage

```python
from backend.core.content_analyzer import UniversalContentAnalyzer, Platform

# Initialize the analyzer
analyzer = UniversalContentAnalyzer(
    openai_api_key="your-api-key",
    anthropic_api_key="optional-anthropic-key",  # Optional
    default_llm="openai"  # or "anthropic"
)

# Analyze content
content = """
Transform your body in just 8 weeks with our revolutionary HIIT program!
No gym required - all workouts can be done at home...
"""

# Run analysis
result = await analyzer.analyze_content(
    content=content,
    platforms=[Platform.INSTAGRAM, Platform.TIKTOK, Platform.YOUTUBE]
)

# Access results
print(f"Business Niche: {result.business_niche.value}")
print(f"Confidence: {result.confidence_score}")
print(f"Target Audience: {result.target_audience}")
print(f"Viral Potential: {result.viral_potential}")
print(f"Recommendations: {result.recommendations}")
```

### Individual Analysis Methods

```python
# Detect business niche only
niche, confidence = await analyzer.detect_business_niche(content)

# Analyze target audience
audience = await analyzer.analyze_target_audience(content)

# Extract brand voice
voice = await analyzer.extract_brand_voice(content)

# Assess viral potential
viral_scores = await analyzer.assess_viral_potential(
    content,
    BusinessNiche.FITNESS_WELLNESS,
    audience,
    [Platform.INSTAGRAM, Platform.TIKTOK]
)
```

## API Reference

### Classes

#### `UniversalContentAnalyzer`

Main class for content analysis.

**Methods:**
- `analyze_content(content, context, platforms)`: Complete content analysis
- `detect_business_niche(content, context)`: Detect business type
- `analyze_target_audience(content, context)`: Identify target audience
- `extract_brand_voice(content, context)`: Extract communication style
- `assess_viral_potential(content, niche, audience, platforms)`: Predict viral potential

#### `ContentAnalysisResult`

Data class containing analysis results.

**Attributes:**
- `business_niche`: Detected business type (BusinessNiche enum)
- `confidence_score`: Confidence in niche detection (0-1)
- `target_audience`: Audience demographics and psychographics
- `brand_voice`: Communication style and preferences
- `viral_potential`: Platform-specific viral scores
- `key_themes`: Main content themes
- `recommendations`: Actionable improvement suggestions
- `metadata`: Analysis metadata

### Enums

#### `BusinessNiche`
- `EDUCATION`: Educational businesses (courses, tutoring)
- `BUSINESS_CONSULTING`: Business consulting and coaching
- `FITNESS_WELLNESS`: Fitness and wellness professionals
- `CREATIVE`: Creative professionals (artists, designers)
- `ECOMMERCE`: E-commerce and retail
- `LOCAL_SERVICE`: Local service businesses
- `TECHNOLOGY`: Tech and SaaS companies
- `NON_PROFIT`: Non-profit organizations
- `OTHER`: Other business types

#### `Platform`
- `INSTAGRAM`
- `TWITTER`
- `LINKEDIN`
- `FACEBOOK`
- `TIKTOK`
- `YOUTUBE`

## Testing

Run the test suite:

```bash
# Run all tests
pytest tests/unit/test_content_analyzer.py

# Run with coverage
pytest tests/unit/test_content_analyzer.py --cov=backend.core.content_analyzer

# Run specific test
pytest tests/unit/test_content_analyzer.py::TestUniversalContentAnalyzer::test_detect_education_niche
```

## Error Handling

The analyzer includes comprehensive error handling:

- **Retry Logic**: Automatically retries failed API calls up to 3 times
- **Graceful Degradation**: Returns partial results when possible
- **Detailed Logging**: All errors are logged with context
- **API Fallback**: Can switch between OpenAI and Anthropic if one fails

## Performance Considerations

- **Parallel Processing**: All analysis tasks run concurrently for speed
- **Content Truncation**: Long content is automatically truncated for API limits
- **Caching**: Consider implementing result caching for repeated content
- **Rate Limiting**: Be aware of API rate limits for your chosen LLM provider

## Examples

### Education Business
```python
education_content = """
Unlock your potential with our Data Science course...
"""
result = await analyzer.analyze_content(education_content)
# Returns: BusinessNiche.EDUCATION with high confidence
```

### Fitness Business
```python
fitness_content = """
Transform your body with our 8-week HIIT program...
"""
result = await analyzer.analyze_content(fitness_content)
# Returns: BusinessNiche.FITNESS_WELLNESS with high confidence
```

### Creative Business
```python
creative_content = """
Stunning photography that tells your story...
"""
result = await analyzer.analyze_content(creative_content)
# Returns: BusinessNiche.CREATIVE with high confidence
```

## Contributing

When adding new features:
1. Ensure they work for ALL business niches
2. Use AI/LLM for logic, never hardcode business rules
3. Include comprehensive tests
4. Update documentation
5. Follow async/await patterns

## License

Part of AutoGuru Universal - See main project license.

---

# AutoGuru Universal - Viral Content Engine

## Overview

The `ViralContentEngine` is an AI-powered viral content generation system that creates platform-optimized social media content with maximum viral potential. It works universally for ANY business niche, using advanced AI strategies to generate engaging content without hardcoded business logic.

## Features

- **Multi-Platform Optimization**: Generates content optimized for Instagram, LinkedIn, TikTok, Twitter, Facebook, YouTube, and Pinterest
- **AI-Driven Viral Strategies**: Uses proven viral patterns adapted to each business niche
- **Smart Hashtag Generation**: Platform-specific hashtag optimization with reach/relevance balance
- **Content Series Creation**: Generate cohesive content series around themes
- **Trending Topic Integration**: Leverage current trends for maximum visibility
- **Optimal Timing Calculation**: Determine best posting times by niche and platform
- **Character Limit Management**: Intelligent content truncation while preserving key messages
- **Comprehensive Error Handling**: Includes retry logic and fallback mechanisms

## Installation

```bash
# Install required dependencies
pip install -r requirements.txt

# Set up environment variables
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"  # Optional
```

## Usage

### Basic Viral Content Generation

```python
from backend.core.viral_engine import ViralContentEngine, BusinessPersona
from backend.models.content_models import Platform

# Initialize the engine
engine = ViralContentEngine(
    openai_api_key="your-api-key",
    anthropic_api_key="optional-anthropic-key",  # Optional
    default_llm="openai"  # or "anthropic"
)

# Create business persona (from content analysis or manually)
persona = BusinessPersona(
    audience_profile=audience_profile,  # From content analyzer
    brand_voice=brand_voice,            # From content analyzer
    business_niche=BusinessNicheType.FITNESS_WELLNESS,
    content_themes=content_themes       # From content analyzer
)

# Generate viral content for multiple platforms
source_content = "5 minute morning workout that boosts energy"
platforms = [Platform.INSTAGRAM, Platform.LINKEDIN, Platform.TIKTOK]

result = await engine.generate_viral_content(
    source_content=source_content,
    persona=persona,
    platforms=platforms
)

# Access platform-specific content
instagram_content = result[Platform.INSTAGRAM]
print(f"Instagram: {instagram_content.content_text}")
print(f"Hashtags: {instagram_content.hashtags}")
print(f"Format: {instagram_content.content_format}")
print(f"CTA: {instagram_content.call_to_action}")
```

### Generate Content Series

```python
# Create a content series around a theme
series = await engine.generate_content_series(
    theme="30-Day Transformation Challenge",
    persona=persona,
    count=7  # Generate 7 pieces
)

for i, content in enumerate(series):
    print(f"Day {i+1}: {content.text}")
    print(f"Format: {content.format}")
    print(f"Viral Score: {content.viral_score}")
```

### Leverage Trending Topics

```python
# Create content based on trending topics
trending_topics = ["New Year Resolutions", "Home Workouts", "Mental Health"]

trending_content = await engine.create_trending_content(
    niche=BusinessNicheType.FITNESS_WELLNESS,
    trending_topics=trending_topics
)

for content in trending_content:
    print(f"Topic: {content.metadata['trending_topic']}")
    print(f"Content: {content.text}")
    print(f"Viral Elements: {content.metadata['viral_elements']}")
```

### Optimize Hashtags

```python
# Generate optimized hashtags for any content
hashtags = await engine.optimize_hashtags(
    content="Transform your morning routine",
    platform=Platform.INSTAGRAM,
    niche=BusinessNicheType.BUSINESS_CONSULTING
)

print(f"Optimized hashtags: {hashtags}")
# Output: ['#morningroutine', '#businesssuccess', '#productivity', ...]
```

### Calculate Optimal Posting Times

```python
# Get best posting times for your audience
posting_times = await engine.calculate_optimal_timing(
    niche=BusinessNicheType.FITNESS_WELLNESS,
    platform=Platform.INSTAGRAM,
    timezone_str="US/Eastern"
)

for time in posting_times:
    print(f"Post at: {time.strftime('%I:%M %p')}")
# Output: Post at: 06:00 AM, 12:00 PM, 05:30 PM, 07:00 PM
```

## API Reference

### Classes

#### `ViralContentEngine`

Main class for viral content generation.

**Methods:**
- `generate_viral_content(source_content, persona, platforms)`: Generate multi-platform viral content
- `optimize_for_platform(content, platform, niche)`: Optimize content for specific platform
- `generate_content_series(theme, persona, count)`: Create themed content series
- `create_trending_content(niche, trending_topics)`: Generate trending topic content
- `optimize_hashtags(content, platform, niche)`: Generate optimized hashtags
- `calculate_optimal_timing(niche, platform, timezone_str)`: Get best posting times

#### `BusinessPersona`

Combines audience profile, brand voice, and business context.

**Attributes:**
- `audience_profile`: Target audience demographics and preferences
- `brand_voice`: Communication style and tone
- `business_niche`: Business category (enum)
- `content_themes`: List of relevant content themes

#### `Content`

Represents generated content with metadata.

**Attributes:**
- `id`: Unique content identifier
- `text`: Content text
- `format`: Content format (image, video, carousel, etc.)
- `platform`: Target platform
- `hashtags`: Optimized hashtags
- `viral_score`: Predicted viral potential (0-1)
- `metadata`: Additional content metadata

### Platform Constraints

The engine respects platform-specific constraints:

| Platform | Max Length | Max Hashtags | Supported Formats |
|----------|------------|--------------|-------------------|
| Instagram | 2,200 chars | 30 | Image, Video, Carousel, Story |
| LinkedIn | 3,000 chars | 5 | Text, Image, Video, Article |
| TikTok | 150 chars | 100 | Video |
| Twitter | 280 chars | 2 | Text, Image, Video |
| Facebook | 63,206 chars | 10 | Text, Image, Video, Live |
| YouTube | 5,000 chars (desc) | N/A | Video |

### Viral Patterns by Niche

The engine uses AI-optimized viral patterns:

- **Fitness/Wellness**: Transformation hooks, before/after formats, motivation triggers
- **Business Consulting**: Success stories, frameworks, FOMO elements
- **Creative**: Behind-the-scenes, process videos, visual inspiration
- **Education**: Quick lessons, myth busting, curiosity gaps

## Testing

Run the comprehensive test suite:

```bash
# Run all viral engine tests
pytest tests/test_viral_engine.py

# Run with coverage
pytest tests/test_viral_engine.py --cov=backend.core.viral_engine

# Run specific test categories
pytest tests/test_viral_engine.py -k "test_generate_viral_content"
pytest tests/test_viral_engine.py -k "test_optimize_hashtags"
```

## Performance Tips

- **Caching**: Hashtags and timing calculations are cached automatically
- **Parallel Processing**: Multiple platforms are optimized concurrently
- **Content Reuse**: Base viral content is generated once and adapted
- **Batch Operations**: Use content series for bulk generation

## Example Outputs

### Fitness Content → Instagram
```
🔥 Transform your mornings in just 5 minutes!

No equipment needed 💪 Here's your energy-boosting routine:

1️⃣ Jump Jacks (30s)
2️⃣ Mountain Climbers (30s)
3️⃣ Burpees (30s)
...

Save this for tomorrow morning! 🌅

#morningroutine #5minuteworkout #homeworkout #fitnessmotivation ...
```

### Business Content → LinkedIn
```
How I Increased Productivity by 300% with This Morning Routine

After analyzing data from 100+ successful entrepreneurs, I discovered...

Key Takeaways:
• Start with intention setting
• Block time for deep work
• Eliminate decision fatigue

What's your morning routine? Share below!

#productivity #entrepreneurship #leadership
```

## Contributing

When adding features to the Viral Content Engine:
1. Ensure universal compatibility across ALL business niches
2. Use AI for content generation, never hardcode templates
3. Test with multiple platforms and niches
4. Maintain platform constraint compliance
5. Add comprehensive error handling

## Integration with Content Analyzer

The Viral Content Engine is designed to work seamlessly with the Content Analyzer:

```python
# Full workflow example
analyzer = UniversalContentAnalyzer(openai_api_key="key")
engine = ViralContentEngine(openai_api_key="key")

# Analyze existing content
analysis = await analyzer.analyze_content(existing_content)

# Create persona from analysis
persona = BusinessPersona(
    audience_profile=analysis.audience_profile,
    brand_voice=analysis.brand_voice,
    business_niche=analysis.business_niche,
    content_themes=analysis.content_themes
)

# Generate viral content
viral_content = await engine.generate_viral_content(
    "New topic to make viral",
    persona,
    [Platform.INSTAGRAM, Platform.TIKTOK]
)
```

## License

Part of AutoGuru Universal - See main project license.