#!/usr/bin/env python3
"""
Text Processing Script using Gemini AI
Analyzes transcript content and removes unnecessary parts like attendance, side conversations, etc.
"""

import os
import re
import json
from typing import List, Dict, Tuple
import google.generativeai as genai
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class TranscriptProcessor:
    def __init__(self, api_key: str = None):
        """Initialize the transcript processor with Gemini API."""
        if api_key:
            genai.configure(api_key=api_key)
        else:
            # Try to get API key from environment variable
            api_key = os.getenv('GEMINI_API_KEY')
            if api_key:
                genai.configure(api_key=api_key)
            else:
                raise ValueError("Gemini API key not provided. Set GEMINI_API_KEY environment variable or pass api_key parameter.")
        
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
    def parse_transcript(self, file_path: str) -> List[Dict]:
        """Parse transcript file and extract timestamped content."""
        transcript_entries = []
        
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Regex pattern to match timestamp format: 00:00:00 - 00:00:03: content
        pattern = r'(\d{2}:\d{2}:\d{2})\s*-\s*(\d{2}:\d{2}:\d{2}):\s*(.+)'
        matches = re.findall(pattern, content)
        
        for match in matches:
            start_time, end_time, text = match
            transcript_entries.append({
                'start_time': start_time,
                'end_time': end_time,
                'text': text.strip(),
                'original_line': f"{start_time} - {end_time}: {text.strip()}"
            })
        
        return transcript_entries
    
    def analyze_transcript_topic(self, transcript_entries: List[Dict]) -> str:
        """Use Gemini to analyze the overall topic of the transcript."""
        # Combine all text content for analysis
        full_text = " ".join([entry['text'] for entry in transcript_entries])
        
        # Limit text length for API call (Gemini has token limits)
        if len(full_text) > 8000:
            full_text = full_text[:8000] + "..."
        
        prompt = f"""
        Analyze the following transcript and identify the main topic or subject matter.
        This appears to be a transcript with timestamps. Please provide a brief, clear description of what this content is about.
        
        Transcript content:
        {full_text}
        
        Please respond with just the main topic in 1-2 sentences.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Error analyzing transcript topic: {e}")
            return "Unable to determine topic"
    
    def identify_unnecessary_content(self, transcript_entries: List[Dict], topic: str) -> List[bool]:
        """Use Gemini to identify which lines contain unnecessary content."""
        unnecessary_flags = []
        
        # Process in batches to avoid token limits
        batch_size = 10
        for i in range(0, len(transcript_entries), batch_size):
            batch = transcript_entries[i:i + batch_size]
            
            # Create context for this batch
            batch_text = "\n".join([
                f"{j+1}. {entry['start_time']} - {entry['end_time']}: {entry['text']}"
                for j, entry in enumerate(batch)
            ])
            
            prompt = f"""
            You are analyzing a transcript about: {topic}
            
            For each numbered line below, determine if it contains unnecessary content that should be removed.
            Be AGGRESSIVE in identifying unnecessary content. Unnecessary content includes:
            - Taking attendance or roll call
            - Side conversations not related to the main topic
            - Technical difficulties or audio issues ("Can everyone hear me?", "Is the microphone working?")
            - Off-topic discussions
            - Repetitive or filler content
            - Administrative announcements
            - Personal conversations unrelated to the main topic
            - Student-to-student conversations about homework, personal matters
            - Any content that doesn't directly contribute to the main educational topic
            
            IMPORTANT: If a line is not directly related to the main topic ({topic}), mark it as unnecessary (true).
            
            Transcript lines:
            {batch_text}
            
            Respond with a JSON array of true/false values, where true means the line is unnecessary and should be replaced with "xxx".
            Example: [false, true, false, true, false]
            """
            
            try:
                response = self.model.generate_content(prompt)
                response_text = response.text.strip()
                
                # Try to parse JSON response
                try:
                    # Clean the response text to extract JSON
                    if '[' in response_text and ']' in response_text:
                        start = response_text.find('[')
                        end = response_text.rfind(']') + 1
                        json_text = response_text[start:end]
                        batch_flags = json.loads(json_text)
                    else:
                        batch_flags = json.loads(response_text)
                    
                    if isinstance(batch_flags, list) and len(batch_flags) == len(batch):
                        unnecessary_flags.extend(batch_flags)
                    else:
                        # Fallback: mark all as necessary if parsing fails
                        unnecessary_flags.extend([False] * len(batch))
                except json.JSONDecodeError as e:
                    # Fallback: mark all as necessary if JSON parsing fails
                    unnecessary_flags.extend([False] * len(batch))
                    
            except Exception as e:
                print(f"Error processing batch {i//batch_size + 1}: {e}")
                # Fallback: mark all as necessary if API call fails
                unnecessary_flags.extend([False] * len(batch))
        
        return unnecessary_flags
    
    def process_transcript(self, input_file: str, output_file: str = None) -> str:
        """Main method to process transcript and create cleaned version."""
        if output_file is None:
            output_file = input_file.replace('.txt', '_cleaned.txt')
        
        print(f"Processing transcript: {input_file}")
        
        # Parse transcript
        transcript_entries = self.parse_transcript(input_file)
        print(f"Found {len(transcript_entries)} transcript entries")
        
        if not transcript_entries:
            print("No transcript entries found. Please check the file format.")
            return output_file
        
        # Analyze topic
        print("Analyzing transcript topic...")
        topic = self.analyze_transcript_topic(transcript_entries)
        print(f"Identified topic: {topic}")
        
        # Identify unnecessary content
        print("Identifying unnecessary content...")
        unnecessary_flags = self.identify_unnecessary_content(transcript_entries, topic)
        
        # Create cleaned transcript
        cleaned_lines = []
        for entry, is_unnecessary in zip(transcript_entries, unnecessary_flags):
            if is_unnecessary:
                cleaned_line = f"{entry['start_time']} - {entry['end_time']}: xxx"
            else:
                cleaned_line = entry['original_line']
            cleaned_lines.append(cleaned_line)
        
        # Write cleaned transcript
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write('\n'.join(cleaned_lines))
        
        # Print summary
        unnecessary_count = sum(unnecessary_flags)
        print(f"Processing complete!")
        print(f"Total lines: {len(transcript_entries)}")
        print(f"Unnecessary lines removed: {unnecessary_count}")
        print(f"Cleaned transcript saved to: {output_file}")
        
        return output_file

def main():
    """Main function to run the transcript processor."""
    # Check for API key (now loaded from .env file)
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("Please set the GEMINI_API_KEY in your .env file.")
        print("You can get an API key from: https://makersuite.google.com/app/apikey")
        return
    
    # Initialize processor
    try:
        processor = TranscriptProcessor()
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Process transcript
    input_file = "transcribed_text.txt"
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Please create the transcript file first.")
        return
    
    try:
        output_file = processor.process_transcript(input_file)
        print(f"\nCleaned transcript saved to: {output_file}")
    except Exception as e:
        print(f"Error processing transcript: {e}")

if __name__ == "__main__":
    main()
