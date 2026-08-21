import os

class TokenManager:
    """
    Helper class for managing authentication tokens (NEUPRINT_TOKEN, CAVE_TOKEN, BANC_TOKEN).
    
    Token Priority (in order of precedence):
    1. Direct input (if provided)
    2. config.json tokens section (committed clean defaults; the file a
       GitHub-pulled copy edits directly)
    3. config_local.json tokens section (gitignored developer-specific
       fallback; only fills entries empty in config.json)
    4. Environment variables
    
    Token Type Detection (for direct input without specifying type):
    - NeuPrint tokens: JWT format, typically 150+ characters with '.' separators
    - CAVE tokens: Short hex strings, typically 32 characters
    """
    
    # Token length thresholds for auto-detection
    NEUPRINT_TOKEN_MIN_LENGTH = 100  # JWT tokens are long
    CAVE_TOKEN_MAX_LENGTH = 64       # CAVE tokens are short hex strings
    
    def __init__(self, project_root=None):
        """project_root overrides auto-detection (used by tests to isolate
        the config files actually read; defaults to the repository root)."""
        self._project_root = project_root
        self.tokens = self._load_tokens_from_files()
        
    def _load_tokens_from_files(self):
        """Load tokens from config.json first, then config_local.json."""
        tokens = {}
        
        # Check current directory and project root; default to the repository
        # root (this file is in src/utils/).
        project_root = self._project_root
        if project_root is None:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        # config.json wins per key (the file a GitHub-pulled copy edits);
        # the gitignored config_local.json only fills empty entries.
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
            # utf-8-sig tolerates the UTF-8 BOM that Windows editors (e.g.
            # Notepad) prepend to JSON files; plain utf-8 would reject it.
            with open(config_path, 'r', encoding='utf-8-sig') as f:
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
                # First non-empty value wins: config.json is parsed first.
                tokens_dict.setdefault(token_key, value.strip())

    def get_token(self, token_name, direct_input=None):
        """
        Get token by name.
        
        Priority order:
        1. Direct input (if provided)
        2. config.json / config_local.json tokens section (config.json wins)
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
            
        # 2. Config files (config.json wins per key)
        file_token = self.tokens.get(token_name)
        if file_token and not file_token.startswith('YOUR_'):
            return file_token
            
        # 3. Environment variable (fallback)
        env_token = os.environ.get(token_name)
        if not env_token and token_name == 'NEUPRINT_TOKEN':
            # The canonical variable neuprint-python itself reads; the
            # legacy NEUPRINT_TOKEN name is still accepted as an alias.
            env_token = os.environ.get('NEUPRINT_APPLICATION_CREDENTIALS')
        if env_token:
            return env_token
            
        return None

    def get_neuprint_token(self):
        """NeuPrint token with the full DROCAT fallback chain.

        Delegates to :meth:`get_token`, which reads the canonical
        ``NEUPRINT_APPLICATION_CREDENTIALS`` variable as an alias of
        ``NEUPRINT_TOKEN``: config.json per key, then the gitignored
        config_local.json, then the environment variables.
        """
        return self.get_token('NEUPRINT_TOKEN')
    
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
