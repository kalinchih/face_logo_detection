import json
import uuid
import boto3
import base64
import logging
import urllib.parse
from botocore.exceptions import ClientError
from google.cloud import vision
from google.oauth2 import service_account
import openai 
from PIL import Image
import base64
import io
import re  

# Initialize Logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize S3 and Rekognition clients
s3_client = boto3.client('s3')
rekognition_client = boto3.client('rekognition')

# Name of the S3 bucket for uploading files
# UPLOAD_BUCKET_NAME = ''
# S3 bucket name for storing known face images
KNOWN_FACES_BUCKET_NAME = ''
# Rekognition Collection ID for storing known face images
KNOWN_FACES_REKOGNITION_COLLECTION_ID = ''
# S3 bucket name for storing known logo images
# KNOWN_LOGOS_BUCKET_NAME = ''
# Default minimum similarity
DEFAULT_MIN_SIMILARITY = 50.0  # Changed to a decimal for precision

# Basic authentication username and password (should be securely stored in environment variables or AWS Secrets Manager)
AUTHORIZED_USERNAME = ''
AUTHORIZED_PASSWORD = ''

# Google Vision API configuration
credentials = service_account.Credentials.from_service_account_file('./service_account_key.json')
vision_client = vision.ImageAnnotatorClient(credentials=credentials)

# OpenAI API Key configuration
OPENAI_API_KEY = ''

def lambda_handler(event, context):
    try:
        logger.info("Lambda function execution started")
        headers = event.get('headers', {})
        logger.info(f"Received headers: {headers}")

        # 1. Check basic authentication
        if not authenticate(event):
            return create_response(401, 'Unauthorized', 'Unauthorized')

        # Get the base64 encoded content of the file from the API Gateway request
        body = event['body']
        if event.get('isBase64Encoded', False):
            file_content = base64.b64decode(body)
            logger.info("File decoded to binary format")
        else:
            file_content = body.encode('utf-8')
            logger.info("File converted to UTF-8 format")

        # Get the Content-Type from HTTP headers to determine the file extension
        content_type = event['headers'].get('content-type', '')
        extension = get_extension_from_content_type(content_type)
        
        if not extension:
            raise ValueError(f"Invalid or unsupported Content-Type: {content_type}")

        # Get the original file name from the query string
        query_params = event.get('queryStringParameters', {})
        original_file_name = query_params.get('fileName', 'uploaded_file')
        original_file_name = urllib.parse.unquote_plus(original_file_name)  # Decode URL-encoded file name
        
        # Split file name and extension
        if '.' in original_file_name:
            base_name, ext = original_file_name.rsplit('.', 1)
            new_file_name = f"{base_name}_{uuid.uuid4()}.{ext}"
        else:
            new_file_name = f"{original_file_name}_{uuid.uuid4()}.{extension}"
        
        logger.info(f"Generated file name with original name and UUID: {new_file_name}")

        '''
        # Upload the file to the S3 bucket for uploading
        try:
            s3_client.put_object(
                Bucket=UPLOAD_BUCKET_NAME,
                Key=new_file_name,
                Body=file_content,
                ContentType=content_type
            )
            logger.info(f"File successfully uploaded to S3: {UPLOAD_BUCKET_NAME}/{new_file_name}")
        except ClientError as s3_error:
            logger.error(f"S3 file upload failed: {str(s3_error)}")
            return create_response(500, 'S3 file upload failed', str(s3_error))
        '''

        # Get minimum similarity from query string, or use the default value if not provided
        min_similarity = query_params.get('minSimilarity')
        if min_similarity:
            try:
                min_similarity = float(min_similarity)
                if not (0.0 <= min_similarity <= 100.0):
                    raise ValueError("minSimilarity must be between 0.0 and 100.0")
            except ValueError:
                logger.warning("Invalid minSimilarity query string, using default value 80.0")
                min_similarity = DEFAULT_MIN_SIMILARITY
        else:
            min_similarity = DEFAULT_MIN_SIMILARITY

        # Get satisfied similarity from query string, or set it to min_similarity if not provided
        satisfied_similarity = query_params.get('satisfiedSimilarity')
        if satisfied_similarity:
            try:
                satisfied_similarity = float(satisfied_similarity)
                if not (0.0 <= satisfied_similarity <= 100.0):
                    raise ValueError("satisfiedSimilarity must be between 0.0 and 100.0")
            except ValueError:
                logger.warning("Invalid satisfiedSimilarity query string, using min_similarity instead")
                satisfied_similarity = min_similarity
        else:
            satisfied_similarity = min_similarity

        logger.info(f"Using minimum similarity threshold: {min_similarity}, satisfied similarity threshold: {satisfied_similarity}")

        try:
            # Detect if the uploaded image contains a face
            if detect_face(file_content, new_file_name):
                # If a face is detected, perform face comparison
                logger.info(f"Face detected in uploaded image {new_file_name}, performing face comparison")

                update_index = query_params.get('updateIndex')
                if update_index == 'face' or update_index == 'all':
                    logger.info("updateIndex is set to face or all, updating rekognition_collection_index")
                    update_face_rekognition_collection_index()
            
                # face_matches = compare_faces_with_known_images(new_file_name, min_similarity, satisfied_similarity)
                face_matches = compare_face_with_rekognition_collection(file_content, min_similarity, satisfied_similarity)
                if check_satisfied_similarity(face_matches, satisfied_similarity):
                    return create_response(200, 'File uploaded and face comparison complete with satisfied similarity', {
                        'toMatchFileName': new_file_name,
                        'faceMatchCount': len(face_matches),
                        'logoMatchCount': 0,
                        'faceMatches': face_matches,
                        'logoMatches': []
                    })
                logo_matches = []
            else:
                # If no face is detected, perform logo comparison
                logger.info(f"No face detected in uploaded image {new_file_name}, performing logo comparison")
                face_matches = []
                logo_matches = compare_logo_with_gcp_vision_api(file_content, min_similarity, satisfied_similarity)

            
                if not logo_matches:  # If logo_matches is empty
                    logger.info("No logos found using GCP Vision API, switching to ChatGPT API")
                    logo_matches = compare_logo_with_chatgpt_api(file_content, min_similarity)
            

                if check_satisfied_similarity(logo_matches, satisfied_similarity):
                    return create_response(200, 'File uploaded and logo comparison complete with satisfied similarity', {
                        'toMatchFileName': new_file_name,
                        'faceMatchCount': 0,
                        'logoMatchCount': len(logo_matches),
                        'faceMatches': [],
                        'logoMatches': logo_matches
                    })
        except ClientError as e:
            logger.error(f"Permission error during detection or comparison: {str(e)}")
            return create_response(403, 'Permission error during detection or comparison', str(e))
        except Exception as e:
            logger.error(f"Unknown error during detection or comparison: {str(e)}")
            return create_response(500, 'Unknown error during detection or comparison', str(e))

        # Count the number of matches
        face_match_count = len(face_matches)
        logo_match_count = len(logo_matches)
        
        # Return the comparison results
        return create_response(200, 'File uploaded and comparison complete', {
            'toMatchFileName': new_file_name,
            'faceMatchCount': face_match_count,  # Return the number of face matches
            'logoMatchCount': logo_match_count,  # Return the number of logo matches
            'faceMatches': face_matches,  # List of face match results
            'logoMatches': logo_matches   # List of logo match results
        })
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return create_response(500, 'File upload and comparison failed', str(e))


def authenticate(event):
    """
    Authenticate username and password for basic authentication
    """
    # Check if 'headers' exist in the event
    if 'headers' not in event:
        logger.warning('Missing headers information')
        return False

    auth_header = event['headers'].get('authorization', '')
    if not auth_header.startswith('Basic '):
        logger.warning('Missing or invalid authorization header')
        return False

    try:
        # Decode basic authentication header
        auth_encoded = auth_header.split(' ')[1]
        auth_decoded = base64.b64decode(auth_encoded).decode('utf-8')
        username, password = auth_decoded.split(':', 1)

        # Check if username and password match
        if username == AUTHORIZED_USERNAME and password == AUTHORIZED_PASSWORD:
            return True
        else:
            logger.warning('Invalid username or password')
            return False
    except Exception as e:
        logger.error(f"Error parsing Authorization header: {str(e)}")
        return False


def get_extension_from_content_type(content_type):
    """
    Get file extension based on Content-Type
    """
    content_type_to_extension = {
        'image/jpeg': 'jpg',
        'image/png': 'png',
        # More Content-Types can be added as needed
    }
    return content_type_to_extension.get(content_type, '')


def detect_face(file_content, file_name):
    """
    Detect if the image contains a face
    """
    try:
        logger.info(f"Checking if the image {file_name} contains a face")
        response = rekognition_client.detect_faces(
            Image={'Bytes': file_content},
            Attributes=['ALL']  # Extract all attributes
        )
        logger.info(f"Face detection result for {file_name}: {response}")  # Add logging to check detection result
        # Return True if a face is found
        return len(response['FaceDetails']) > 0
    except ClientError as e:
        logger.error(f"Permission error during face detection: {str(e)}", exc_info=True)
        raise e
    except Exception as e:
        logger.error(f"Error during face detection: {str(e)}", exc_info=True)
        raise Exception(f"Error in detect_face(): {str(e)}")


def update_face_rekognition_collection_index():
    try:
        setup_rekognition_collection(KNOWN_FACES_REKOGNITION_COLLECTION_ID)
        # List all objects in the S3 Bucket
        response = s3_client.list_objects_v2(Bucket=KNOWN_FACES_BUCKET_NAME)
        
        # Check if there is any content
        if 'Contents' not in response:
            logger.info(f"No images in S3 bucket {KNOWN_FACES_BUCKET_NAME}")
            return

        # Iterate through each object in the S3 bucket
        for obj in response['Contents']:
            image_key = obj['Key']  # Get the file name
            
            logger.info(f"Adding image {image_key} to Rekognition Collection")

            try:
                # Add the S3 image to the Rekognition Collection
                rekognition_client.index_faces(
                    CollectionId=KNOWN_FACES_REKOGNITION_COLLECTION_ID,
                    Image={
                        'S3Object': {
                            'Bucket': KNOWN_FACES_BUCKET_NAME,
                            'Name': image_key
                        }
                    },
                    ExternalImageId=image_key,  
                    DetectionAttributes=['ALL']  # Optional: Detect all attributes
                )
                logger.info(f"Successfully added {image_key} to Rekognition Collection {KNOWN_FACES_REKOGNITION_COLLECTION_ID}")

            except ClientError as e:
                logger.error(f"Error adding image {image_key} to Rekognition Collection {KNOWN_FACES_REKOGNITION_COLLECTION_ID}: {str(e)}", exc_info=True)

    except ClientError as e:
        logger.error(f"Error listing S3 bucket {KNOWN_FACES_BUCKET_NAME}: {str(e)}", exc_info=True)
        raise Exception(f"Error in update_face_rekognition_collection_index(): {str(e)}")


def setup_rekognition_collection(rekognition_collection_id):
    try:
        delete_rekognition_collection(rekognition_collection_id)
        rekognition_client.create_collection(CollectionId=rekognition_collection_id)
        logger.info(f"Successfully created Rekognition Collection: {rekognition_collection_id}")
    except Exception as e:
        logger.error(f"Error creating Rekognition Collection: {str(e)}", exc_info=True)
        raise Exception(f"Error in setup_rekognition_collection(): {str(e)}")


def delete_rekognition_collection(rekognition_collection_id):
    try:
        if check_collection_exists(rekognition_collection_id):
            response = rekognition_client.delete_collection(CollectionId=rekognition_collection_id)
            if response['StatusCode'] == 200:
                logger.info(f"Successfully deleted Rekognition Collection: {rekognition_collection_id}")
            else:
                logger.error(f"Failed to delete Rekognition Collection {rekognition_collection_id}, status code: {response['StatusCode']}")
    except Exception as e:
        logger.error(f"Error deleting Rekognition Collection: {str(e)}")
        raise Exception(f"Error in delete_rekognition_collection(): {str(e)}")


def check_collection_exists(rekognition_collection_id):
    """Check if the Rekognition Collection exists"""
    try:
        rekognition_client.describe_collection(CollectionId=rekognition_collection_id)
        logger.info(f"Rekognition Collection exists: {rekognition_collection_id}")
        return True
    except rekognition_client.exceptions.ResourceNotFoundException:
        logger.info(f"Rekognition Collection does not exist: {rekognition_collection_id}")
        return False


def compare_face_with_rekognition_collection(file_content, min_similarity, satisfied_similarity):
    """
    Perform face comparison using Rekognition Collection
    """
    try:
        # Perform face comparison using Rekognition
        matches = rekognition_client.search_faces_by_image(
            CollectionId=KNOWN_FACES_REKOGNITION_COLLECTION_ID,
            Image={'Bytes': file_content},
            MaxFaces=3,
            FaceMatchThreshold=min_similarity  # Use default similarity threshold
        ).get('FaceMatches', [])
        
        face_matches = []

        # If there are matches, add them to the result list
        if matches: 
            for match in matches:
                similarity = match.get('Similarity', 0)  # Use get method to avoid errors
                if similarity >= min_similarity:
                    # Correctly extract ExternalImageId and remove commas
                    externalImageId = match['Face'].get('ExternalImageId', 'Unknown')

                    # Only split if externalImageId exists and is not 'Unknown'
                    if externalImageId != 'Unknown':
                        brand = externalImageId.split('-')[0].strip()
                    else:
                        brand = 'Unknown'

                    face_matches.append({
                        'brand': brand,
                        'similarity': similarity,
                        'knownImage': externalImageId,
                        'boundingBox': match['Face']['BoundingBox'],
                        'type': 'FACE'  # Mark comparison type as face
                    })
                    logger.info(f"Found similar face: Similarity {similarity}%, Image {externalImageId}")
                    
                    # Stop comparison if similarity reaches satisfied similarity
                    if similarity >= satisfied_similarity:
                        logger.info(f"Satisfied similarity {similarity}%, stopping comparison")
                        return face_matches

        return face_matches

    except Exception as e:
        logger.error(f"Error during face comparison: {str(e)}", exc_info=True)
        raise Exception(f"Error in compare_face_with_rekognition_collection(): {str(e)}")


def compare_logo_with_gcp_vision_api(uploaded_file_content, min_similarity, satisfied_similarity):
    """
    Compare logos using Google Vision API's logo detection
    """
    try:
        # Use Google Vision API to detect logos
        image = vision.Image(content=uploaded_file_content)
        response = vision_client.logo_detection(image=image)

        logos = response.logo_annotations
        logger.info(f"Number of logos detected by Google Vision API: {len(logos)}")

        logo_matches = []
        for logo in logos:
            similarity = logo.score * 100
            if similarity >= min_similarity:
                # Convert boundingPoly to a dictionary
                bounding_poly_dict = {
                    'vertices': [{'x': vertex.x, 'y': vertex.y} for vertex in logo.bounding_poly.vertices]
                }
                
                logo_matches.append({
                    'brand': logo.description,
                    'similarity': similarity,
                    'knownImage': 'Unknown',
                    'boundingBox': bounding_poly_dict,  # Use dictionary instead of BoundingPoly object
                    'type': 'LOGO'
                })
                logger.info(f"Found logo: {logo.description}, Similarity: {similarity}%")

                # Stop comparison if similarity reaches satisfied similarity
                if (similarity * 100) >= satisfied_similarity:
                    logger.info(f"Satisfied similarity {similarity}%, stopping comparison")
                    return logo_matches
        return logo_matches
    except Exception as e:
        logger.error(f"Error in Google Vision API logo detection: {str(e)}", exc_info=True)
        raise Exception(f"Error in compare_logo_with_gcp_vision_api(): {str(e)}")


from PIL import Image
import io
import base64
import logging

logger = logging.getLogger()

def compress_image_to_base64(file_content, target_scale=0.1):
    """
    Compress the image and convert it to Base64 encoding
    :param file_content: Original image content (bytes)
    :param target_scale: Compression ratio (0-1), here the image is resized to 10% of its original size
    :return: Compressed Base64 encoded image
    """
    try:
        # Convert bytes to image object
        img = Image.open(io.BytesIO(file_content))

        # Determine the image format, JPG and PNG formats support compression
        img_format = img.format.lower()

        # If the image is in RGBA mode (with transparency), convert it to RGB when saving as JPG
        if img.mode == 'RGBA' and img_format == 'jpeg':
            img = img.convert('RGB')

        # Resize the image to the target_scale ratio
        new_size = (int(img.width * target_scale), int(img.height * target_scale))
        img = img.resize(new_size, Image.ANTIALIAS)

        # Create a byte stream to save the compressed image
        buffer = io.BytesIO()

        # Save compressed image based on its format
        if img_format == 'jpeg' or img_format == 'jpg':
            # Save as JPEG format with quality set to 85
            img.save(buffer, format="JPEG", quality=85)
        elif img_format == 'png':
            # Save as PNG format, compressing while retaining transparency
            img.save(buffer, format="PNG", optimize=True)

        # Convert the byte stream to Base64 encoding
        base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return base64_image
    except Exception as e:
        logger.error(f"Error compressing image: {str(e)}")
        raise Exception(f"Error in compress_image_to_base64(): {str(e)}")


def compare_logo_with_chatgpt_api(file_content, min_similarity):
    """
    Use the new OpenAI API for logo detection and return logo description and similarity
    """
    try:
        # Compress image and convert it to base64 encoding (resize to 10% of the original size)
        image_base64 = compress_image_to_base64(file_content, target_scale=0.1)

        # Set the payload for the API request
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Which \"Taiwanese brand logo\" is this image? "
                                # "Provide a structured JSON response with 3 fields: content, 'brand' and 'similarity'. "
                                # "'content' for ChatCompletionMessage.content, "
                                # "'brand' for the brand name and 'similarity' for the confidence percentage. "
                                #"If the brand is not identifiable, set 'brand' to 'Unknown' and 'similarity' to 0. "
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300
        }

        client = openai.Client(api_key=OPENAI_API_KEY)

        # Make the request using the new chat.completions.create() method
        response = client.chat.completions.create(**payload)

        logger.info(f"Full response from OpenAI API: {response}")

        brand = 'Unknown'
        similarity = 95.0
        try:
            # Parse the response from OpenAI, expecting it to be in JSON format
            result_text = response.choices[0].message.content.strip()
            brand = extract_logo_name(result_text)
            
        except Exception as e:
            logger.error(f"Error parsing response from OpenAI API: {str(e)}")

        if brand == 'Unknown':
            similarity = 20.0
    
        logo_matches = [{
            'brand': brand,
            'similarity': similarity,
            'knownImage': 'Unknown',
            'gptResponse': result_text,  
            'type': 'LOGO'
        }]

        if similarity < min_similarity:
            logo_matches = []
        
        return logo_matches

    except Exception as e:
        logger.error(f"Error during logo comparison using OpenAI API: {str(e)}", exc_info=True)
        raise Exception(f"Error in compare_logo_with_chatgpt_api(): {str(e)}")


# Define the function to extract logo names
def extract_logo_name(text):
    def clean_match(match):
        """Clean up the matched string, remove trailing commas/periods, 'the' at the beginning, and trim whitespace"""
        return match.group(1).rstrip(',.').lstrip("the").strip()

    # can't identify
    if re.search(r'can\'t identify', text):
        return "Unknown"

    # unable to identify
    if re.search(r'unable to identify', text):
        return "Unknown"

    # Match brand names within double or single quotes
    match = re.search(r'[\'"]([^\'"]+)[\'"]', text)
    if match:
        return clean_match(match)

    # Match brand names within ** characters
    match = re.search(r'\*\*(.*?)\*\*', text)
    if match:
        return clean_match(match)

    # Match brand names without quotes
    patterns = [
        r'belongs to ([\w\s\(\)]+)\,', 
        r'belongs to ([\w\s\(\)]+)\.', 
        r'associated with ([\w\s\(\)]+)\.',
        r'associated with ([\w\s\(\)]+)\,',
        r'is for ([\w\s\(\)]+)\,', 
        r'is for ([\w\s\(\)]+)\.', 
        r'brand ([\w\s\(\)]+)\.',
        r'brand ([\w\s\(\)]+)\,',
        r'logo of ([\w\s\(\)]+)\.', 
        r'logo of ([\w\s\(\)]+)\,',
        r'company ([\w\s\(\)]+)\.',
        r'company ([\w\s\(\)]+)\,',
        r'organization ([\w\s\(\)]+)\.',
        r'organization ([\w\s\(\)]+)\,',
        r'department ([\w\s\(\)]+)\.',
        r'department ([\w\s\(\)]+)\,'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return clean_match(match)

    return "Unknown"



def check_satisfied_similarity(matches, satisfied_similarity):
    """
    Check if any match has reached the satisfied similarity threshold
    """
    for match in matches:
        # Ensure the key exists before accessing, to avoid KeyError
        similarity = match.get('similarity', 0)
        confidence = match.get('confidence', 0)
        if similarity >= satisfied_similarity or confidence >= satisfied_similarity:
            return True
    return False


def create_response(status_code, message, data):
    """
    Create a standardized API response format
    """
    return {
        'statusCode': status_code,
        'body': json.dumps({
            'message': message,
            'data': data
        })
    }