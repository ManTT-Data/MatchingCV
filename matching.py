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

class MatchingCV:
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Template để matching CV với JD
        self.matching_template = """
        You are an AI Recruiter Assistant.

        Your task is to analyze how well a candidate CV matches a job description (JD) based on 10 key features.

        Below is the candidate CV (in JSON format) and the JD (free text format). Your goal is to:
        - Score each feature individually on a scale of 0–100
        - Justify each score briefly (1–2 sentences)
        - Calculate the final weighted matching score (0–100) based on the following weights:

        Scoring weights:
        - exp_years: 20%
        - prof_skill_advanced: 18%
        - soft_skill: 14%
        - education: 13%
        - prof_skill_basic: 10%
        - achievements: 8%
        - relevant_projects: 7%
        - certs: 5%
        - language: 4%
        - activities: 1%

        Scoring Guidelines:
        - exp_years: Compare candidate's years of experience with JD requirements
        - prof_skill_advanced: Match advanced technical skills mentioned in JD
        - soft_skill: Evaluate communication, leadership, teamwork skills
        - education: Compare degree level and field relevance
        - prof_skill_basic: Match basic/fundamental skills required
        - achievements: Assess awards, recognitions, notable accomplishments
        - relevant_projects: Evaluate project experience relevance
        - certs: Match certifications with job requirements
        - language: Assess language proficiency requirements. If JD doesn't mention languages, give a default score of 100.
        - activities: Consider extracurricular activities relevance

        ---
        ### 📄 CV (JSON format):
        {CV_JSON_HERE}

        ---
        ### 📝 Job Description (JD):
        {JD_TEXT_HERE}

        ---
        ### 🎯 Output ONLY valid JSON in this exact format:
        {{
        "scores": {{
            "exp_years": {{
            "score": 0,
            "justification": "Brief explanation here"
            }},
            "prof_skill_advanced": {{
            "score": 0,
            "justification": "Brief explanation here"
            }},
            "soft_skill": {{
            "score": 0,
            "justification": "Brief explanation here"
            }},
            "education": {{
            "score": 0,
            "justification": "Brief explanation here"
            }},
            "prof_skill_basic": {{
            "score": 0,
            "justification": "Brief explanation here"
            }},
            "achievements": {{
            "score": 0,
            "justification": "Brief explanation here"
            }},
            "relevant_projects": {{
            "score": 0,
            "justification": "Brief explanation here"
            }},
            "certs": {{
            "score": 0,
            "justification": "Brief explanation here"
            }},
            "language": {{
            "score": 0,
            "justification": "Brief explanation here"
            }},
            "activities": {{
            "score": 0,
            "justification": "Brief explanation here"
            }}
        }},
        "final_matching_score": 0.0
        }}
        """

    def calculate_matching_score(self, cv_json, jd_text):
        """Tính toán matching score giữa CV và JD"""
        try:
            # Format prompt với CV JSON và JD text
            prompt = self.matching_template.format(
                CV_JSON_HERE=json.dumps(cv_json, ensure_ascii=False),
                JD_TEXT_HERE=jd_text
            )
            
            # Generate response using OpenAI
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert recruiter AI. Always respond with valid JSON only. No additional text or formatting."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=3000
            )
            
            # Extract text from response
            result_text = response.choices[0].message.content
            
            # Clean up the response text
            if result_text.startswith('```json'):
                result_text = result_text.replace('```json', '').replace('```', '').strip()
            elif result_text.startswith('```'):
                result_text = result_text.replace('```', '').strip()
            
            # Parse JSON response
            matching_result = json.loads(result_text)
            return matching_result
            
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            print(f"Raw response: {result_text}")
            return None
        except Exception as e:
            print(f"Error calculating matching score: {e}")
            return None

    def process_matching(self, cv_json_path, jd_text_path):
        """Process matching từ file CV JSON và JD text"""
        try:
            # Load CV JSON
            with open(cv_json_path, "r", encoding="utf-8") as f:
                cv_json = json.load(f)
            
            # Load JD text
            with open(jd_text_path, "r", encoding="utf-8") as f:
                jd_text = f.read()
            
            print("Files loaded successfully")
            print(f"CV keys: {list(cv_json.keys())}")
            print(f"JD text length: {len(jd_text)} characters")
            
            # Calculate matching score
            matching_result = self.calculate_matching_score(cv_json, jd_text)
            
            if matching_result:
                print("Matching score calculated successfully")
                return matching_result
            else:
                print("Failed to calculate matching score")
                return None
                
        except FileNotFoundError as e:
            print(f"File not found: {e}")
            return None
        except Exception as e:
            print(f"Error processing matching: {e}")
            return None

    def save_matching_result(self, matching_result, output_path):
        """Save matching result to JSON file"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(matching_result, f, ensure_ascii=False, indent=2)
            print(f"Matching result saved to: {output_path}")
        except Exception as e:
            print(f"Error saving matching result: {e}")

    def display_matching_summary(self, matching_result):
        """Display a formatted summary of matching results"""
        if not matching_result:
            print("No matching result to display")
            return
        
        print("\n" + "="*60)
        print("            CV-JD MATCHING SUMMARY")
        print("="*60)
        
        scores = matching_result.get("scores", {})
        
        # Define weights for calculation verification
        weights = {
            "exp_years": 0.20,
            "prof_skill_advanced": 0.18,
            "soft_skill": 0.14,
            "education": 0.13,
            "prof_skill_basic": 0.10,
            "achievements": 0.08,
            "relevant_projects": 0.07,
            "certs": 0.05,
            "language": 0.04,
            "activities": 0.01
        }
        
        print(f"📊 FINAL MATCHING SCORE: {matching_result.get('final_matching_score', 0):.1f}/100")
        print("\n🔍 DETAILED BREAKDOWN:")
        print("-" * 60)
        
        for feature, weight in weights.items():
            if feature in scores:
                score_info = scores[feature]
                score = score_info.get("score", 0)
                justification = score_info.get("justification", "No justification provided")
                
                print(f"• {feature.replace('_', ' ').title()}: {score}/100 (Weight: {weight*100}%)")
                print(f"  └─ {justification}")
                print()

class JDLoader:
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
    
    def save_loaded_info(self, jd_info, output_path):
        """Save extracted information to txt file"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(jd_info)
            print(f"JD information saved to: {output_path}")
        except Exception as e:
            print(f"Error saving JD info: {e}")

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
    cv_extractor = CVExtractor()
    pdf_path = "cv3.pdf"
    output_path = "Loaded_Info/extracted_cv_info.json"
    
    cv_info = cv_extractor.process_cv(pdf_path)
    if cv_info:
        cv_extractor.save_extracted_info(cv_info, output_path)
    else:
        print("Failed to extract CV information.")
    
    # Example usage for JDLoader
    jd_loader = JDLoader()
    jd_pdf_path = "job_description.pdf"
    jd_info = jd_loader.load_pdf(jd_pdf_path)
    if jd_info:
        jd_loader.save_extracted_info(jd_info, "Loaded_Info/extracted_jd_info.txt")
        print("Job description information extracted and saved successfully.")
    else:
        print("Failed to extract job description information.")
    
    # Example usage for MatchingCV
    matching_cv = MatchingCV()
    
    # Process matching using saved files
    matching_result = matching_cv.process_matching(
        cv_json_path="extracted_cv_info.json",
        jd_text_path="extracted_jd_info.txt"
    )
    
    if matching_result:
        # Save matching result
        matching_cv.save_matching_result(matching_result, "Loaded_Info/matching_result.json")

        # Display summary
        matching_cv.display_matching_summary(matching_result)
    else:
        print("Failed to calculate matching score.")

