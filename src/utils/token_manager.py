import os

class TokenManager:
    """
    Helper class for managing authentication tokens (NEUPRINT_TOKEN, CAVE_TOKEN, BANC_TOKEN).
    
    Token Priority (in order of precedence):
    1. Direct input (if provided)
    2. config_local.json tokens section (gitignored local override)
    3. config.json tokens section (committed clean defaults)
    4. Environment variables
    
    Token Type Detection (for direct input without specifying type):
    - NeuPrint tokens: JWT format, typically 150+ characters with '.' separators
    - CAVE tokens: Short hex strings, typically 32 characters
    """
    
    # Token length thresholds for auto-detection
    NEUPRINT_TOKEN_MIN_LENGTH = 100  # JWT tokens are long
    CAVE_TOKEN_MAX_LENGTH = 64       # CAVE tokens are short hex strings
    
    def __init__(self):
        self.tokens = self._load_tokens_from_files()
        
    def _load_tokens_from_files(self):
        """Load tokens from config_local.json (override) then config.json."""
        tokens = {}
        
        # Check current directory and project root
        # Assuming this file is in src/utils/
        # Project root is ../../ relative to this file
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        # config.json ships clean on GitHub; the gitignored config_local.json
        # overrides it per key (non-empty values only).
        self._parse_config_file(project_root, 'config.json', tokens)
        self._parse_config_file(project_root, 'config_local.json', tokens)
            
        return tokens

    def _parse_config_file(self, project_root, filename, tokens_dict):
        """Load the tokens section of one project config file."""
        import json
        search_paths = [
            filename, # Current working directory
            os.path.join(project_root, filename) # Project root
        ]
        config_path = None
        for path in search_paths:
            if os.path.exists(path):
                config_path = path
                break
        if not config_path:
            return
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to read {filename}: {e}")
            return
        section = data.get('tokens') if isinstance(data, dict) else None
        if not isinstance(section, dict):
            return
        for config_key, token_key in (
                ('neuprint', 'NEUPRINT_TOKEN'),
                ('cave', 'CAVE_TOKEN')):
            value = section.get(config_key)
            if isinstance(value, str) and value.strip():
                tokens_dict[token_key] = value.strip()

    def get_token(self, token_name, direct_input=None):
        """
        Get token by name.
        
        Priority order:
        1. Direct input (if provided)
        2. config_local.json / config.json tokens section
        3. Environment variable
        
        Args:
            token_name (str): Name of the token (e.g., 'NEUPRINT_TOKEN', 'CAVE_TOKEN')
            direct_input (str, optional): Directly provided token.
            
        Returns:
            str: The token value, or None if not found.
        """
        # 1. Direct input
        if direct_input:
            return direct_input
            
        # 2. Local file (highest priority after direct input)
        file_token = self.tokens.get(token_name)
        if file_token and not file_token.startswith('YOUR_'):
            return file_token
            
        # 3. Environment variable (fallback)
        env_token = os.environ.get(token_name)
        if env_token:
            return env_token
            
        return None
    
    def detect_token_type(self, token):
        """
        Detect token type based on format/length.
        
        Args:
            token (str): The token string
            
        Returns:
            str: 'neuprint', 'cave', or 'unknown'
        """
        if not token:
            return 'unknown'
        
        token_len = len(token)
        
        # NeuPrint tokens are JWTs - long with '.' separators
        if token_len >= self.NEUPRINT_TOKEN_MIN_LENGTH and '.' in token:
            return 'neuprint'
        
        # CAVE tokens are short hex strings (typically 32 chars)
        if token_len <= self.CAVE_TOKEN_MAX_LENGTH:
            # Check if it's hex-like
            if all(c in '0123456789abcdefABCDEF' for c in token):
                return 'cave'
        
        return 'unknown'
    
    def get_auto_token(self, direct_input=None, prefer_type=None):
        """
        Get token with auto-detection of token type.
        
        If direct_input is provided, detect its type and return it for the appropriate use.
        If both NEUPRINT and CAVE tokens are needed, raises a notice.
        
        Args:
            direct_input (str, optional): Directly provided token
            prefer_type (str, optional): Preferred token type ('neuprint' or 'cave')
            
        Returns:
            dict: {'neuprint': token_or_none, 'cave': token_or_none, 'detected_type': str}
        """
        result = {
            'neuprint': None,
            'cave': None,
            'detected_type': None
        }
        
        if direct_input:
            detected = self.detect_token_type(direct_input)
            result['detected_type'] = detected
            
            if detected == 'neuprint':
                result['neuprint'] = direct_input
                # Also check for CAVE token in files
                result['cave'] = self.get_token('CAVE_TOKEN')
            elif detected == 'cave':
                result['cave'] = direct_input
                # Also check for NeuPrint token in files
                result['neuprint'] = self.get_token('NEUPRINT_TOKEN')
            else:
                # Unknown type - use as-is based on prefer_type
                if prefer_type == 'neuprint':
                    result['neuprint'] = direct_input
                elif prefer_type == 'cave':
                    result['cave'] = direct_input
        else:
            # No direct input - get both from files/env
            result['neuprint'] = self.get_token('NEUPRINT_TOKEN')
            result['cave'] = self.get_token('CAVE_TOKEN')
        
        return result
    
    def require_both_tokens(self, direct_input=None):
        """
        Get both tokens, raising ValueError if one is missing.
        
        Args:
            direct_input (str, optional): Directly provided token
            
        Returns:
            dict: {'neuprint': token, 'cave': token}
            
        Raises:
            ValueError: If both tokens are needed but one is missing
        """
        result = self.get_auto_token(direct_input)
        
        missing = []
        if not result['neuprint']:
            missing.append('NEUPRINT_TOKEN')
        if not result['cave']:
            missing.append('CAVE_TOKEN')
        
        if missing:
            raise ValueError(
                f"Missing required token(s): {', '.join(missing)}.\n"
                f"Please set them in config.json or as environment variables.\n"
                f"Get NEUPRINT_TOKEN from: https://neuprint.janelia.org/account\n"
                f"Get CAVE_TOKEN from: https://codex.flywire.ai/auth_token"
            )
        
        return result

# Singleton instance
token_manager = TokenManager()
