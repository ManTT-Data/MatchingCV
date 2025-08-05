import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai
import json

# Load environment variables
load_dotenv()

# Configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

class CVExtractor:
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Template để extract thông tin CV
        self.extraction_template = """
        ### OVERALL CONTEXT
        You are an expert HR data structuring bot. Your task is to analyze a comprehensive block of text containing multiple sections from a single resume and extract a full, structured JSON object with all required metrics.

        ### EVALUATION PRINCIPLES
        - **Experience:** Calculate the total years of relevant work experience.
        - **Language:** All information about non-native language levels
        - **Education:** All information about educational background, including degrees obtained and institutions attended.
        - **Certs: ** All certifications about courses like coursera, licenses, or professional qualifications.
        - **Achievements: ** All achievements, awards, or recognitions received.
        - **Relevant Projects:** All projects that are relevant to the job position, including roles and contributions.
        - **Activities:** All extracurricular activities or community involvement.
        ### TASK
        Analyze the multi-part input text block below. Respond ONLY with a single, valid JSON object containing all the specified keys. If a section is empty or not applicable, return a default value of 0.

        The required JSON output structure is:
        {{
        "exp_years": <number>,
        "language": <text>,
        "education": <text>,
        "prof_skill_advanced": <text>,
        "prof_skill_basic": <text>,
        "soft_skill": <text>,
        "certs": <text>,
        "achievements": <text>,
        "relevant_projects": <text>,
        "activities": <text>
        }}

        ### YOUR TASK
        Input Text Block:
        ---BEGIN RESUME DATA---
        {cv_text}
        ---END RESUME DATA---

        Your JSON Output:
        """

    def load_pdf(self, pdf_path):
        """Load PDF và extract text"""
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            # Combine all pages
            full_text = ""
            for doc in documents:
                full_text += doc.page_content + "\n"
            
            return full_text
        except Exception as e:
            print(f"Error loading PDF: {e}")
            return None

    def extract_cv_info(self, cv_text):
        """Extract thông tin quan trọng từ CV text"""
        try:
            # Format prompt với CV text
            prompt = self.extraction_template.format(cv_text=cv_text)
            
            # Generate response using OpenAI
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Hoặc "gpt-4" nếu bạn có access
                messages=[
                    {"role": "system", "content": "You are an expert HR data structuring bot. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=2000
            )
            
            # Extract text from response
            result_text = response.choices[0].message.content
            
            # Clean up the response text (remove any markdown formatting)
            if result_text.startswith('```json'):
                result_text = result_text.replace('```json', '').replace('```', '').strip()
            elif result_text.startswith('```'):
                result_text = result_text.replace('```', '').strip()
            
            # Parse JSON response
            cv_info = json.loads(result_text)
            return cv_info
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            print(f"Raw response: {result_text}")
            return None
        except Exception as e:
            print(f"Error extracting CV info: {e}")
            return None

    def process_cv(self, pdf_path):
        """Process CV từ PDF file"""
        print(f"Processing CV: {pdf_path}")
        
        # Load PDF
        cv_text = self.load_pdf(pdf_path)
        if not cv_text:
            return None
        
        print("PDF loaded successfully")
        print(f"Text length: {len(cv_text)} characters")
        
        # Extract information
        cv_info = self.extract_cv_info(cv_text)
        if not cv_info:
            return None
        
        print("CV information extracted successfully")
        return cv_info

    def save_extracted_info(self, cv_info, output_path):
        """Save extracted information to JSON file"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(cv_info, f, ensure_ascii=False, indent=2)
            print(f"CV information saved to: {output_path}")
        except Exception as e:
            print(f"Error saving CV info: {e}")

# Example usage
if __name__ == "__main__":
    extractor = CVExtractor()
    
    # Process CV
    pdf_path = "cv2.pdf"  # Thay đổi path của bạn
    cv_info = extractor.process_cv(pdf_path)
    
    if cv_info:
        print("\n=== EXTRACTED CV INFORMATION ===")
        print(json.dumps(cv_info, ensure_ascii=False, indent=2))
        
        # Save to file
        extractor.save_extracted_info(cv_info, "extracted_cv_info.json")
    else:
        print("Failed to process CV")