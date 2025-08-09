import requests
from typing import Dict, Optional, Tuple

class GoogleFactCheck:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        
    def check_claim(self, text: str) -> Tuple[str, float, Optional[Dict], str]:
        try:
            params = {
                'key': self.api_key,
                'query': text
            }
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'claims' in data and data['claims']:
                claim = data['claims'][0]
                rating = claim.get('textualRating', '').lower()
                
                # Define indicators for fake and true ratings
                false_indicators = ['false', 'fake', 'incorrect', 'misleading', 'pants on fire']
                true_indicators = ['true', 'correct', 'accurate', 'fact']
                
                if any(indicator in rating for indicator in false_indicators):
                    return 'fake', 1.0, claim, 'fact_check'
                elif any(indicator in rating for indicator in true_indicators):
                    return 'real', 1.0, claim, 'fact_check'
                else:
                    # If rating doesn't match any indicators, return as model verification
                    return 'real', 0.6, claim, 'fact_check'
            
            # No fact checks found
            return 'unverified', 0.5, None, 'model'
            
        except Exception as e:
            print(f"Error checking fact: {e}")
            return 'unverified', 0.5, None, 'model'
            