import requests
from typing import Dict, Optional, Tuple

class GoogleFactCheck:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = 'https://factchecktools.googleapis.com/v1alpha1/claims:search'
    
    def check_claim(self, text: str) -> Tuple[str, float, Optional[Dict]]:
        """Check a claim using Google's Fact Check API
        
        Args:
            text: The text/claim to verify
            
        Returns:
            Tuple containing:
            - credibility label ('real'/'fake')
            - confidence score (0-1)
            - fact check details (or None if no matches found)
        """
        try:
            # Query the Fact Check API
            params = {
                'key': self.api_key,
                'query': text,
                'languageCode': 'en'
            }
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Process results
            if 'claims' in data and data['claims']:
                # Get the most relevant fact check
                claim = data['claims'][0]
                
                # Extract the review rating
                review = claim.get('claimReview', [{}])[0]
                rating = review.get('textualRating', '').lower()
                
                # Map the rating to our binary classification
                false_indicators = ['false', 'fake', 'pants on fire', 'incorrect', 
                                  'misleading', 'inaccurate', 'mostly false']
                true_indicators = ['true', 'correct', 'accurate', 'mostly true']
                
                if any(indicator in rating for indicator in false_indicators):
                    label = 'fake'
                    score = 0.9  # High confidence for false claims
                elif any(indicator in rating for indicator in true_indicators):
                    label = 'real'
                    score = 0.9  # High confidence for true claims
                else:
                    label = 'real'  # Default to real for mixed/unclear ratings
                    score = 0.6  # Lower confidence
                
                return label, score, claim
            
            # No fact checks found
            return 'real', 0.5, None
            
        except Exception as e:
            print(f"Error checking fact: {e}")
            # Return neutral result on error
            return 'real', 0.5, None