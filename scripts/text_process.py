#!/usr/bin/env python3
"""
Text Processing Script using Gemini AI
Analyzes transcript content and removes unnecessary parts like attendance, side conversations, etc.
"""

import os
import re
import json
import time
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
        
        # Use Gemini 2.5 Flash - latest model with better performance
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
    def _make_api_call_with_retry(self, prompt: str, max_retries: int = 3) -> str:
        """Make API call with retry logic for quota limits."""
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                error_str = str(e)
                if "429" in error_str and "quota" in error_str.lower():
                    if attempt < max_retries - 1:
                        # Extract retry delay if available
                        retry_delay = 60  # Default 60 seconds
                        if "retry_delay" in error_str:
                            try:
                                # Try to extract seconds from the error message
                                import re
                                delay_match = re.search(r'seconds:\s*(\d+)', error_str)
                                if delay_match:
                                    retry_delay = int(delay_match.group(1)) + 5  # Add 5 seconds buffer
                            except:
                                pass
                        
                        print(f"Quota exceeded. Waiting {retry_delay} seconds before retry {attempt + 1}/{max_retries}...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        print(f"Max retries reached. Skipping this batch.")
                        return None
                else:
                    print(f"API error: {e}")
                    return None
        return None
        
    def parse_srt(self, file_path: str) -> List[Dict]:
        """Parse SRT file and extract timestamped content."""
        transcript_entries = []
        
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Split SRT into blocks
        blocks = content.strip().split('\n\n')
        
        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) >= 3:
                # Extract subtitle number, timestamp, and text
                subtitle_num = lines[0].strip()
                timestamp_line = lines[1].strip()
                text = '\n'.join(lines[2:]).strip()
                
                # Parse timestamp: 00:00:00,000 --> 00:00:05,000
                if ' --> ' in timestamp_line:
                    start_time, end_time = timestamp_line.split(' --> ')
                    transcript_entries.append({
                        'subtitle_num': subtitle_num,
                        'start_time': start_time,
                        'end_time': end_time,
                        'text': text,
                        'duration': self._calculate_duration(start_time, end_time)
                    })
        
        return transcript_entries
    
    def _calculate_duration(self, start_time: str, end_time: str) -> float:
        """Calculate duration in seconds from SRT timestamps."""
        def srt_to_seconds(srt_time):
            # Convert HH:MM:SS,mmm to seconds
            time_part, ms_part = srt_time.split(',')
            h, m, s = map(int, time_part.split(':'))
            ms = int(ms_part)
            return h * 3600 + m * 60 + s + ms / 1000.0
        
        return srt_to_seconds(end_time) - srt_to_seconds(start_time)
    
    def parse_transcript(self, file_path: str) -> List[Dict]:
        """Parse SRT transcript file and extract timestamped content."""
        transcript_entries = []
        
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Split content into subtitle blocks
        blocks = content.strip().split('\n\n')
        
        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) >= 3:
                # Skip the subtitle number (first line)
                # Parse timestamp line (second line)
                timestamp_line = lines[1]
                # Parse text content (remaining lines)
                text_content = '\n'.join(lines[2:])
                
                # Parse SRT timestamp format: 00:00:00,000 --> 00:00:03,000
                timestamp_pattern = r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})'
                match = re.match(timestamp_pattern, timestamp_line)
                
                if match:
                    start_time_srt, end_time_srt = match.groups()
                    # Convert SRT format to simple format for consistency
                    start_time = start_time_srt.replace(',', '.')
                    end_time = end_time_srt.replace(',', '.')
                    
                    transcript_entries.append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'text': text_content.strip(),
                        'original_line': f"{start_time} - {end_time}: {text_content.strip()}"
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
        
        response_text = self._make_api_call_with_retry(prompt)
        if response_text:
            return response_text
        else:
            print("Error analyzing transcript topic: API call failed")
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
            output_file = input_file.replace('.srt', '_cleaned.srt')
        
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
        
        # Create cleaned SRT transcript
        cleaned_srt_blocks = []
        subtitle_number = 1
        
        for entry, is_unnecessary in zip(transcript_entries, unnecessary_flags):
            # Convert back to SRT timestamp format
            start_srt = entry['start_time'].replace('.', ',')
            end_srt = entry['end_time'].replace('.', ',')
            
            if is_unnecessary:
                text_content = "xxx"
            else:
                text_content = entry['text']
            
            # Create SRT block
            srt_block = f"{subtitle_number}\n{start_srt} --> {end_srt}\n{text_content}"
            cleaned_srt_blocks.append(srt_block)
            subtitle_number += 1
        
        # Write cleaned SRT transcript
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write('\n\n'.join(cleaned_srt_blocks))
        
        # Print summary
        unnecessary_count = sum(unnecessary_flags)
        print(f"Processing complete!")
        print(f"Total lines: {len(transcript_entries)}")
        print(f"Unnecessary lines removed: {unnecessary_count}")
        print(f"Cleaned transcript saved to: {output_file}")
        
        return output_file
    
    def condense_to_25_percent(self, transcript_entries: List[Dict], topic: str) -> List[Dict]:
        """Use Gemini to identify coherent segments representing 25% of the most important content."""
        # Calculate target duration (25% of total)
        total_duration = sum(entry['duration'] for entry in transcript_entries)
        target_duration = total_duration * 0.25
        
        print(f"Total video duration: {total_duration:.2f} seconds")
        print(f"Target condensed duration: {target_duration:.2f} seconds (25%)")
        
        # Step 1: Score all segments for importance
        scored_entries = self._score_segments_for_importance(transcript_entries, topic)
        
        # Step 2: Identify coherent segments (sequences of important content)
        coherent_segments = self._identify_coherent_segments(scored_entries, target_duration)
        
        # Step 3: Select the best coherent segments to reach 25% duration
        final_segments = self._select_best_coherent_segments(coherent_segments, target_duration)
        
        print(f"Selected {len(final_segments)} coherent segments")
        print(f"Final duration: {sum(s['duration'] for s in final_segments):.2f} seconds ({(sum(s['duration'] for s in final_segments)/total_duration)*100:.1f}%)")
        
        return final_segments
    
    def _score_segments_for_importance(self, transcript_entries: List[Dict], topic: str) -> List[Dict]:
        """Score each segment for importance while maintaining original order."""
        scored_entries = transcript_entries.copy()
        
        # Process in batches to avoid token limits
        batch_size = 15
        for i in range(0, len(scored_entries), batch_size):
            batch = scored_entries[i:i + batch_size]
            
            # Create context for this batch
            batch_text = "\n".join([
                f"{j+1}. [{entry['start_time']} --> {entry['end_time']}] {entry['text']}"
                for j, entry in enumerate(batch)
            ])
            
            prompt = f"""
            You are analyzing a transcript about: {topic}
            
            Rate each segment's importance for creating a coherent, educational summary.
            Focus on segments that contain complete thoughts and key information.
            
            Score 1-10 where:
            - 9-10: CRITICAL - Core concepts, main points, essential information
            - 7-8: VERY IMPORTANT - Key supporting information, important details
            - 5-6: IMPORTANT - Good supporting content, useful context
            - 3-4: SOMEWHAT USEFUL - Minor details, examples
            - 1-2: UNNECESSARY - Filler, repetition, tangential content
            
            Prioritize segments that:
            - Contain complete thoughts or concepts
            - Introduce important topics or conclusions
            - Provide essential context or explanations
            - Would help maintain narrative flow
            
            Transcript segments:
            {batch_text}
            
            Respond with only a JSON array of scores (1-10).
            Example: [8, 3, 9, 5, 10, 2, 7]
            """
            
            response_text = self._make_api_call_with_retry(prompt)
            
            if response_text:
                # Parse JSON response
                try:
                    if '[' in response_text and ']' in response_text:
                        start = response_text.find('[')
                        end = response_text.rfind(']') + 1
                        json_text = response_text[start:end]
                        importance_scores = json.loads(json_text)
                    else:
                        importance_scores = json.loads(response_text)
                    
                    if isinstance(importance_scores, list) and len(importance_scores) == len(batch):
                        for entry, score in zip(batch, importance_scores):
                            entry['importance_score'] = score
                    else:
                        for entry in batch:
                            entry['importance_score'] = 5
                            
                except json.JSONDecodeError:
                    print(f"Failed to parse JSON response for batch {i//batch_size + 1}")
                    for entry in batch:
                        entry['importance_score'] = 5
            else:
                print(f"No response from API for batch {i//batch_size + 1}")
                for entry in batch:
                    entry['importance_score'] = 5
        
        return scored_entries
    
    def _identify_coherent_segments(self, scored_entries: List[Dict], target_duration: float) -> List[Dict]:
        """Group consecutive high-scoring segments into coherent blocks."""
        coherent_segments = []
        current_segment = None
        min_importance_threshold = 6  # Only consider segments with score >= 6
        
        for entry in scored_entries:
            if entry['importance_score'] >= min_importance_threshold:
                if current_segment is None:
                    # Start new coherent segment
                    current_segment = {
                        'entries': [entry],
                        'start_time': entry['start_time'],
                        'end_time': entry['end_time'],
                        'duration': entry['duration'],
                        'avg_importance': entry['importance_score'],
                        'text_preview': entry['text'][:100] + '...' if len(entry['text']) > 100 else entry['text']
                    }
                else:
                    # Extend current segment
                    current_segment['entries'].append(entry)
                    current_segment['end_time'] = entry['end_time']
                    current_segment['duration'] += entry['duration']
                    current_segment['avg_importance'] = sum(e['importance_score'] for e in current_segment['entries']) / len(current_segment['entries'])
                    
            else:
                # Low importance segment - end current segment if it exists and is substantial
                if current_segment and current_segment['duration'] >= 3.0:  # At least 3 seconds
                    coherent_segments.append(current_segment)
                current_segment = None
        
        # Don't forget the last segment
        if current_segment and current_segment['duration'] >= 3.0:
            coherent_segments.append(current_segment)
        
        # Sort by average importance
        coherent_segments.sort(key=lambda x: x['avg_importance'], reverse=True)
        
        print(f"Identified {len(coherent_segments)} coherent segments")
        return coherent_segments
    
    def _select_best_coherent_segments(self, coherent_segments: List[Dict], target_duration: float) -> List[Dict]:
        """Select the best coherent segments to reach target duration while maintaining flow."""
        selected_segments = []
        accumulated_duration = 0
        
        # First pass: select segments that fit within target
        for segment in coherent_segments:
            if accumulated_duration + segment['duration'] <= target_duration:
                selected_segments.extend(segment['entries'])
                accumulated_duration += segment['duration']
                print(f"Added coherent segment: {segment['text_preview']} (Duration: {segment['duration']:.1f}s, Avg Score: {segment['avg_importance']:.1f})")
            elif accumulated_duration < target_duration * 0.85:  # Allow some flexibility if we're under 85%
                selected_segments.extend(segment['entries'])
                accumulated_duration += segment['duration']
                print(f"Added coherent segment (flex): {segment['text_preview']} (Duration: {segment['duration']:.1f}s)")
        
        # Sort back to chronological order for seamless playback
        selected_segments.sort(key=lambda x: int(x['subtitle_num']))
        
        return selected_segments
    
    def process_srt_file(self, input_file: str, output_file: str = None) -> str:
        """Main method to process SRT file and create condensed version with 25% of content."""
        if output_file is None:
            output_file = input_file.replace('.srt', '_condensed.srt')
        
        print(f"Processing SRT file: {input_file}")
        
        # Parse SRT
        transcript_entries = self.parse_srt(input_file)
        print(f"Found {len(transcript_entries)} SRT entries")
        
        if not transcript_entries:
            print("No SRT entries found. Please check the file format.")
            return output_file
        
        # Analyze topic
        print("Analyzing transcript topic...")
        topic = self.analyze_transcript_topic(transcript_entries)
        print(f"Identified topic: {topic}")
        
        # Condense to 25% of most important content
        print("Condensing to 25% of most important content...")
        condensed_entries = self.condense_to_25_percent(transcript_entries, topic)
        
        # Write condensed SRT file
        self._write_srt_file(condensed_entries, output_file)
        
        print(f"Condensed SRT saved to: {output_file}")
        return output_file
    
    def _write_srt_file(self, entries: List[Dict], output_file: str):
        """Write entries to SRT format file with transition markers for time gaps."""
        with open(output_file, 'w', encoding='utf-8') as file:
            subtitle_counter = 1
            
            for i, entry in enumerate(entries):
                # Check for significant time gap with previous entry
                if i > 0:
                    prev_end_time = self._srt_time_to_seconds(entries[i-1]['end_time'])
                    curr_start_time = self._srt_time_to_seconds(entry['start_time'])
                    time_gap = curr_start_time - prev_end_time
                    
                    # If gap is more than 30 seconds, add a transition marker
                    if time_gap > 30:
                        # Add transition marker subtitle
                        file.write(f"{subtitle_counter}\n")
                        
                        # Use previous end time + 1 second for transition start
                        transition_start = self._seconds_to_srt_time(prev_end_time + 1)
                        transition_end = self._seconds_to_srt_time(prev_end_time + 3)
                        
                        file.write(f"{transition_start} --> {transition_end}\n")
                        
                        # Create descriptive transition text
                        minutes_skipped = int(time_gap // 60)
                        if minutes_skipped >= 1:
                            file.write(f"[...{minutes_skipped} minute{'s' if minutes_skipped != 1 else ''} later...]\n\n")
                        else:
                            file.write(f"[...content omitted...]\n\n")
                        
                        subtitle_counter += 1
                
                # Write the actual subtitle entry
                file.write(f"{subtitle_counter}\n")
                file.write(f"{entry['start_time']} --> {entry['end_time']}\n")
                file.write(f"{entry['text']}\n\n")
                subtitle_counter += 1
    
    def _srt_time_to_seconds(self, srt_time: str) -> float:
        """Convert SRT time format (HH:MM:SS,mmm) to seconds."""
        time_part, ms_part = srt_time.split(',')
        h, m, s = map(int, time_part.split(':'))
        ms = int(ms_part)
        return h * 3600 + m * 60 + s + ms / 1000.0
    
    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT time format (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"

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
    
    # Process SRT file with improved algorithm
    input_file = "../output/transcript.srt"
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Please run transcription first.")
        return
    
    try:
        output_file = processor.process_srt_file(input_file, "../output/transcript_coherent.srt")
        print(f"\nCoherent condensed transcript saved to: {output_file}")
    except Exception as e:
        print(f"Error processing transcript: {e}")

if __name__ == "__main__":
    main()
