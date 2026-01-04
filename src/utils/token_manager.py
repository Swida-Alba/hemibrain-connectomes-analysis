import os
import re

class TokenManager:
    """
    Helper class for managing authentication tokens (NEUPRINT_TOKEN, CAVE_TOKEN).
    
    Priorities:
    1. Direct input (if provided)
    2. Environment variables
    3. Local token_info_local.txt file (gitignored)
    4. Local token_info.txt file
    """
    
    def __init__(self):
        self.tokens = self._load_tokens_from_files()
        
    def _load_tokens_from_files(self):
        """Load tokens from local files."""
        tokens = {}
        
        # Check current directory and project root
        # Assuming this file is in src/utils/
        # Project root is ../../ relative to this file
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Files to check in order (later overrides earlier)
        # 1. token_info.txt (template/defaults)
        # 2. token_info_local.txt (user secrets)
        filenames = ['token_info.txt', 'token_info_local.txt']
        
        for filename in filenames:
            search_paths = [
                filename, # Current working directory
                os.path.join(project_root, filename) # Project root
            ]
            
            file_path = None
            for path in search_paths:
                if os.path.exists(path):
                    file_path = path
                    break
            
            if file_path:
                self._parse_file(file_path, tokens)
            
        return tokens

    def _parse_file(self, file_path, tokens_dict):
        """Parse a token file and update the tokens dictionary."""
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # Parse KEY=VALUE
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Remove quotes if present
                        if (value.startswith("'") and value.endswith("'")) or \
                           (value.startswith('"') and value.endswith('"')):
                            value = value[1:-1]
                            
                        tokens_dict[key] = value
        except Exception as e:
            print(f"Warning: Failed to read token file {file_path}: {e}")

    def get_token(self, token_name, direct_input=None):
        """
        Get token by name.
        
        Args:
            token_name (str): Name of the token (e.g., 'NEUPRINT_TOKEN', 'CAVE_TOKEN')
            direct_input (str, optional): Directly provided token.
            
        Returns:
            str: The token value, or None if not found.
        """
        # 1. Direct input
        if direct_input:
            return direct_input
            
        # 2. Environment variable
        env_token = os.environ.get(token_name)
        if env_token:
            return env_token
            
        # 3. Local file
        return self.tokens.get(token_name)

# Singleton instance
token_manager = TokenManager()
