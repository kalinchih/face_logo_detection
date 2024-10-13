# face_logo_detection
An AWS Lambda Web API to detect face and logo in an uploaded image.

## Architecture
<img width="827" alt="architecture" src="https://github.com/user-attachments/assets/baf2b4e6-a4a1-4626-bf78-34679575b014">


## Face (available via AWS Rekognition):
Currently, AWS Rekognition is very powerful for facial recognition. You can create an index through Rekognition Collection to greatly speed up the comparison.

## Logo (via waterfall checking):
- Google Vision API: Can identify global corporate logos.
- Use the ChatGPT-4o-mini model through ChatGPT OpenAI API to compensate for Google Vision API, but there will be a 10% false positive.
- AWS Rekognition Label and Custom Labels are ineffective and should not be considered for use.
- Will try AWS Bedrock to replace ChatGPT OpenAI API

## Optimal QueryString configuration for this AWS Lambda API:
- satisfiedSimilarity: 90
- minSimilarity: 50
