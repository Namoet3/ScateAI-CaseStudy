import React, { useState, useEffect, useCallback, useRef, createContext, useContext } from 'react';
import { BrowserRouter, Routes, Route, Navigate, useNavigate, useLocation, Link } from 'react-router-dom';

// API Configuration - Use relative URLs so Vite proxy handles them
const API_BASE_URL = '';

// ============================================================================
// Auth Context
// ============================================================================

const AuthContext = createContext(null);

function useAuth() {
  return useContext(AuthContext);
}

function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(() => localStorage.getItem('authToken'));
  const [loading, setLoading] = useState(true);

  // Check if user is logged in on mount
  useEffect(() => {
    if (token) {
      fetchUserInfo(token);
    } else {
      setLoading(false);
    }
  }, []);

  const fetchUserInfo = async (authToken) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/auth/me`, {
        headers: {
          'Authorization': `Bearer ${authToken}`
        }
      });

      if (response.ok) {
        const userData = await response.json();
        setUser(userData);
      } else {
        // Token invalid, clear it
        localStorage.removeItem('authToken');
        setToken(null);
      }
    } catch (error) {
      console.error('Auth check failed:', error);
      localStorage.removeItem('authToken');
      setToken(null);
    } finally {
      setLoading(false);
    }
  };

  const login = async (email, password) => {
    const response = await fetch(`${API_BASE_URL}/api/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password })
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Login failed');
    }

    const data = await response.json();
    localStorage.setItem('authToken', data.access_token);
    setToken(data.access_token);
    setUser(data.user);
    return data;
  };

  const signup = async (email, password, name) => {
    const response = await fetch(`${API_BASE_URL}/api/auth/signup`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password, name })
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Signup failed');
    }

    const data = await response.json();
    localStorage.setItem('authToken', data.access_token);
    setToken(data.access_token);
    setUser(data.user);
    return data;
  };

  const logout = () => {
    localStorage.removeItem('authToken');
    setToken(null);
    setUser(null);
  };

  return (
    <AuthContext.Provider value={{ user, token, loading, login, signup, logout, isAuthenticated: !!user }}>
      {children}
    </AuthContext.Provider>
  );
}

// ============================================================================
// Custom Hooks
// ============================================================================

function useApi() {
  const auth = useAuth();

  const fetchApi = useCallback(async (endpoint, options = {}) => {
    const headers = {
      ...options.headers,
    };

    // Add auth token if available
    if (auth?.token) {
      headers['Authorization'] = `Bearer ${auth.token}`;
    }

    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      ...options,
      headers,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'API error' }));
      throw new Error(error.detail || 'API error');
    }

    return response.json();
  }, [auth?.token]);

  return { fetchApi };
}

function useJobPolling(jobId, onComplete, onLyricsReceived) {
  const { fetchApi } = useApi();
  const [status, setStatus] = useState(null);
  const completedRef = useRef(false);

  // Reset completed flag when jobId changes
  useEffect(() => {
    completedRef.current = false;
  }, [jobId]);

  useEffect(() => {
    if (!jobId) return;

    const pollInterval = setInterval(async () => {
      try {
        const data = await fetchApi(`/api/jobs/${jobId}`);
        setStatus(data);

        // If lyrics are available, send them back
        if (data.lyrics && onLyricsReceived) {
          onLyricsReceived(data.lyrics);
        }

        if ((data.status === 'completed' || data.status === 'failed') && !completedRef.current) {
          completedRef.current = true;
          clearInterval(pollInterval);
          if (onComplete) onComplete(data);
        }
      } catch (error) {
        console.error('Polling error:', error);
      }
    }, 2000);

    return () => clearInterval(pollInterval);
  }, [jobId, fetchApi]);

  return status;
}

// Animated progress hook
function useAnimatedProgress(isActive, realProgress = null) {
  const [progress, setProgress] = useState(0);
  const intervalRef = useRef(null);

  useEffect(() => {
    if (isActive) {
      setProgress(0);
      let currentProgress = 0;

      intervalRef.current = setInterval(() => {
        // Slow down as we approach 90%
        const increment = currentProgress < 30 ? 3 :
          currentProgress < 60 ? 2 :
            currentProgress < 85 ? 1 : 0.2;

        currentProgress = Math.min(currentProgress + increment, 95);
        setProgress(Math.floor(currentProgress));
      }, 300);

      return () => {
        if (intervalRef.current) clearInterval(intervalRef.current);
      };
    } else {
      if (intervalRef.current) clearInterval(intervalRef.current);
      if (realProgress === 100) {
        setProgress(100);
      }
    }
  }, [isActive, realProgress]);

  return progress;
}

// ============================================================================
// Components
// ============================================================================

// ============================================================================
// Auth Modal Component
// ============================================================================

function AuthModal({ isOpen, onClose, initialMode = 'login' }) {
  const [mode, setMode] = useState(initialMode); // 'login' or 'signup'
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [name, setName] = useState('');

  // Reset mode when modal opens
  useEffect(() => {
    if (isOpen) {
      setMode(initialMode);
    }
  }, [isOpen, initialMode]);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const auth = useAuth();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      if (mode === 'login') {
        await auth.login(email, password);
      } else {
        await auth.signup(email, password, name);
      }
      onClose();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <>
      <div
        style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'rgba(0,0,0,0.7)',
          zIndex: 1000,
        }}
        onClick={onClose}
      />
      <div style={{
        position: 'fixed',
        top: '50%',
        left: '50%',
        transform: 'translate(-50%, -50%)',
        background: '#1a1a2e',
        borderRadius: '16px',
        padding: '32px',
        width: '100%',
        maxWidth: '400px',
        zIndex: 1001,
        border: '1px solid rgba(255,255,255,0.1)',
      }}>
        <h2 style={{ margin: '0 0 24px 0', color: '#fff', textAlign: 'center' }}>
          {mode === 'login' ? '👋 Welcome Back' : '🎵 Create Account'}
        </h2>

        {error && (
          <div style={{
            padding: '12px',
            background: 'rgba(239, 68, 68, 0.2)',
            border: '1px solid rgba(239, 68, 68, 0.3)',
            borderRadius: '8px',
            color: '#fca5a5',
            marginBottom: '16px',
            fontSize: '0.9rem',
          }}>
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit}>
          {mode === 'signup' && (
            <div style={{ marginBottom: '16px' }}>
              <label style={{ display: 'block', marginBottom: '6px', color: '#aaa', fontSize: '0.9rem' }}>
                Name
              </label>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Your name"
                style={{
                  width: '100%',
                  padding: '12px',
                  background: 'rgba(255,255,255,0.05)',
                  border: '1px solid rgba(255,255,255,0.1)',
                  borderRadius: '8px',
                  color: '#fff',
                  fontSize: '1rem',
                }}
              />
            </div>
          )}

          <div style={{ marginBottom: '16px' }}>
            <label style={{ display: 'block', marginBottom: '6px', color: '#aaa', fontSize: '0.9rem' }}>
              Email
            </label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@example.com"
              required
              style={{
                width: '100%',
                padding: '12px',
                background: 'rgba(255,255,255,0.05)',
                border: '1px solid rgba(255,255,255,0.1)',
                borderRadius: '8px',
                color: '#fff',
                fontSize: '1rem',
              }}
            />
          </div>

          <div style={{ marginBottom: '24px' }}>
            <label style={{ display: 'block', marginBottom: '6px', color: '#aaa', fontSize: '0.9rem' }}>
              Password
            </label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••"
              required
              minLength={6}
              style={{
                width: '100%',
                padding: '12px',
                background: 'rgba(255,255,255,0.05)',
                border: '1px solid rgba(255,255,255,0.1)',
                borderRadius: '8px',
                color: '#fff',
                fontSize: '1rem',
              }}
            />
          </div>

          <button
            type="submit"
            disabled={loading}
            style={{
              width: '100%',
              padding: '14px',
              background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)',
              border: 'none',
              borderRadius: '8px',
              color: '#fff',
              fontSize: '1rem',
              fontWeight: '600',
              cursor: loading ? 'not-allowed' : 'pointer',
              opacity: loading ? 0.7 : 1,
            }}
          >
            {loading ? '⏳ Please wait...' : (mode === 'login' ? 'Log In' : 'Sign Up')}
          </button>
        </form>

        <div style={{ marginTop: '20px', textAlign: 'center' }}>
          <span style={{ color: '#666' }}>
            {mode === 'login' ? "Don't have an account? " : "Already have an account? "}
          </span>
          <button
            onClick={() => { setMode(mode === 'login' ? 'signup' : 'login'); setError(''); }}
            style={{
              background: 'none',
              border: 'none',
              color: '#818cf8',
              cursor: 'pointer',
              fontSize: '1rem',
            }}
          >
            {mode === 'login' ? 'Sign Up' : 'Log In'}
          </button>
        </div>

        <button
          onClick={onClose}
          style={{
            position: 'absolute',
            top: '12px',
            right: '12px',
            background: 'none',
            border: 'none',
            color: '#666',
            fontSize: '24px',
            cursor: 'pointer',
          }}
        >
          ×
        </button>
      </div>
    </>
  );
}

function Header() {
  const auth = useAuth();
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [authMode, setAuthMode] = useState('login');

  const openAuth = (mode) => {
    setAuthMode(mode);
    setShowAuthModal(true);
  };

  return (
    <>
      <header style={styles.header}>
        <div style={styles.headerContent}>
          <h1 style={styles.logo}>🎵 AI Music Generator</h1>
          <p style={styles.tagline}>Create original music and voice covers with AI</p>
        </div>

        <div style={{ position: 'absolute', top: '20px', right: '20px' }}>
          {auth?.isAuthenticated ? (
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <span style={{ color: '#aaa', fontSize: '0.9rem' }}>
                👤 {auth.user?.name || auth.user?.email}
              </span>
              <button
                onClick={auth.logout}
                style={{
                  padding: '8px 16px',
                  background: 'rgba(255,255,255,0.1)',
                  border: '1px solid rgba(255,255,255,0.2)',
                  borderRadius: '6px',
                  color: '#fff',
                  cursor: 'pointer',
                  fontSize: '0.85rem',
                }}
              >
                Log Out
              </button>
            </div>
          ) : (
            <div style={{ display: 'flex', gap: '10px' }}>
              <Link
                to="/login"
                style={{
                  padding: '10px 20px',
                  background: 'rgba(255,255,255,0.1)',
                  border: '1px solid rgba(255,255,255,0.2)',
                  borderRadius: '8px',
                  color: '#fff',
                  cursor: 'pointer',
                  fontWeight: '600',
                  fontSize: '0.9rem',
                  textDecoration: 'none',
                }}
              >
                Log In
              </Link>
              <Link
                to="/signup"
                style={{
                  padding: '10px 20px',
                  background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)',
                  border: 'none',
                  borderRadius: '8px',
                  color: '#fff',
                  cursor: 'pointer',
                  fontWeight: '600',
                  fontSize: '0.9rem',
                  textDecoration: 'none',
                }}
              >
                Sign Up
              </Link>
            </div>
          )}
        </div>
      </header>

      <AuthModal isOpen={showAuthModal} onClose={() => setShowAuthModal(false)} initialMode={authMode} />
    </>
  );
}

function TabNav() {
  const location = useLocation();
  const tabs = [
    { id: 'music', path: '/music', label: '🎼 Original Music', flow: 'Flow A' },
    { id: 'cover', path: '/cover', label: '🎤 Voice Cover', flow: 'Flow B' },
    { id: 'combined', path: '/combined', label: '✨ Combined', flow: 'Flow C' },
    { id: 'train', path: '/voice-models', label: '🎭 Voice Models', flow: 'Custom' },
  ];

  const getActiveTab = () => {
    const path = location.pathname;
    const tab = tabs.find(t => t.path === path);
    return tab ? tab.id : 'music';
  };

  const activeTab = getActiveTab();

  return (
    <nav style={styles.tabNav}>
      {tabs.map(tab => (
        <Link
          key={tab.id}
          to={tab.path}
          style={{
            ...styles.tab,
            ...(activeTab === tab.id ? styles.tabActive : {}),
            textDecoration: 'none'
          }}
        >
          <span style={styles.tabLabel}>{tab.label}</span>
          <span style={styles.tabFlow}>{tab.flow}</span>
        </Link>
      ))}
    </nav>
  );
}

// Music generation options
const MUSIC_OPTIONS = {
  moods: [
    { value: '', label: 'Select Mood...' },
    { value: 'happy', label: '😊 Happy & Uplifting' },
    { value: 'sad', label: '😢 Sad & Melancholic' },
    { value: 'energetic', label: '⚡ Energetic & Powerful' },
    { value: 'calm', label: '😌 Calm & Peaceful' },
    { value: 'romantic', label: '💕 Romantic & Dreamy' },
    { value: 'angry', label: '😤 Angry & Intense' },
    { value: 'mysterious', label: '🌙 Mysterious & Dark' },
    { value: 'nostalgic', label: '🕰️ Nostalgic & Retro' },
    { value: 'epic', label: '🏔️ Epic & Cinematic' },
    { value: 'playful', label: '🎪 Playful & Fun' },
  ],
  genres: [
    { value: '', label: 'Select Genre...' },
    { value: 'pop', label: '🎤 Pop' },
    { value: 'rock', label: '🎸 Rock' },
    { value: 'hiphop', label: '🎧 Hip Hop / Rap' },
    { value: 'rnb', label: '🎹 R&B / Soul' },
    { value: 'electronic', label: '🎛️ Electronic / EDM' },
    { value: 'jazz', label: '🎷 Jazz' },
    { value: 'classical', label: '🎻 Classical' },
    { value: 'country', label: '🤠 Country' },
    { value: 'indie', label: '🌿 Indie / Alternative' },
    { value: 'metal', label: '🤘 Metal' },
    { value: 'lofi', label: '📻 Lo-Fi' },
    { value: 'latin', label: '💃 Latin' },
    { value: 'kpop', label: '🇰🇷 K-Pop' },
    { value: 'reggae', label: '🌴 Reggae' },
  ],
  themes: [
    { value: '', label: 'Select Theme...' },
    { value: 'love', label: '❤️ Love & Relationships' },
    { value: 'party', label: '🎉 Party & Celebration' },
    { value: 'motivation', label: '💪 Motivation & Success' },
    { value: 'heartbreak', label: '💔 Heartbreak & Loss' },
    { value: 'adventure', label: '🗺️ Adventure & Travel' },
    { value: 'nature', label: '🌿 Nature & Environment' },
    { value: 'night', label: '🌃 Night Life & City' },
    { value: 'summer', label: '☀️ Summer Vibes' },
    { value: 'winter', label: '❄️ Winter & Holidays' },
    { value: 'freedom', label: '🦅 Freedom & Independence' },
    { value: 'reflection', label: '🪞 Self-Reflection' },
  ],
  styles: [
    { value: '', label: 'Select Style...' },
    { value: 'modern', label: '✨ Modern & Trendy' },
    { value: 'vintage', label: '📼 Vintage & Retro' },
    { value: 'acoustic', label: '🪕 Acoustic & Unplugged' },
    { value: 'orchestral', label: '🎼 Orchestral & Grand' },
    { value: 'minimalist', label: '⬜ Minimalist & Simple' },
    { value: 'experimental', label: '🔬 Experimental & Avant-garde' },
    { value: 'polished', label: '💎 Polished & Radio-ready' },
    { value: 'raw', label: '🔥 Raw & Authentic' },
    { value: 'dreamy', label: '☁️ Dreamy & Atmospheric' },
    { value: 'aggressive', label: '⚔️ Aggressive & Hard-hitting' },
  ]
};

function PromptSelector({ onSelect }) {
  const [prompts, setPrompts] = useState({});
  const { fetchApi } = useApi();

  useEffect(() => {
    fetchApi('/api/prompts')
      .then(data => setPrompts(data.prompts || {}))
      .catch(() => {
        // Use fallback prompts if API fails
        setPrompts({
          early_2000s_pop: { name: "Early 2000s Pop Rock", prompt: "Catchy early 2000s pop rock anthem, energetic female vocalist..." },
          indie_rock: { name: "Dreamy Indie Rock", prompt: "Dreamy, psychedelic, slow Indie Rock, reverb-soaked vocals..." },
          lofi: { name: "Lo-Fi Hip Hop", prompt: "Chill lo-fi hip hop beat, dusty vinyl texture..." },
        });
      });
  }, [fetchApi]);

  return (
    <div style={styles.promptSelector}>
      <label style={styles.label}>Quick Prompts:</label>
      <div style={styles.promptGrid}>
        {Object.entries(prompts).map(([key, value]) => (
          <button
            key={key}
            type="button"
            onClick={() => onSelect(value.prompt)}
            style={styles.promptButton}
          >
            {value.name}
          </button>
        ))}
      </div>
    </div>
  );
}

function MusicGenerationForm({ onSubmit, loading, generatedLyrics, onLyricsChange }) {
  const [prompt, setPrompt] = useState('');
  const [duration, setDuration] = useState(60);
  const [instrumental, setInstrumental] = useState(false);
  const [lyrics, setLyrics] = useState('');
  const [showHelp, setShowHelp] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Dropdown states
  const [mood, setMood] = useState('');
  const [genre, setGenre] = useState('');
  const [theme, setTheme] = useState('');
  const [style, setStyle] = useState('');

  // Build hidden prompt from selections
  const buildHiddenPrompt = () => {
    const parts = [];
    if (mood) {
      const moodLabel = MUSIC_OPTIONS.moods.find(m => m.value === mood)?.label.replace(/^[^\s]+\s/, '') || mood;
      parts.push(moodLabel.toLowerCase() + ' mood');
    }
    if (genre) {
      const genreLabel = MUSIC_OPTIONS.genres.find(g => g.value === genre)?.label.replace(/^[^\s]+\s/, '') || genre;
      parts.push(genreLabel + ' genre');
    }
    if (theme) {
      const themeLabel = MUSIC_OPTIONS.themes.find(t => t.value === theme)?.label.replace(/^[^\s]+\s/, '') || theme;
      parts.push(`theme of ${themeLabel.toLowerCase()}`);
    }
    if (style) {
      const styleLabel = MUSIC_OPTIONS.styles.find(s => s.value === style)?.label.replace(/^[^\s]+\s/, '') || style;
      parts.push(`${styleLabel.toLowerCase()} production style`);
    }
    return parts.length > 0 ? parts.join(', ') : '';
  };

  // Update lyrics when generated lyrics are received
  useEffect(() => {
    if (generatedLyrics && !lyrics) {
      setLyrics(generatedLyrics);
      if (onLyricsChange) onLyricsChange(generatedLyrics);
    }
  }, [generatedLyrics]);

  const handleSubmit = (e) => {
    e.preventDefault();
    const hiddenPrompt = buildHiddenPrompt();
    const fullPrompt = [prompt, hiddenPrompt].filter(Boolean).join('. ');

    onSubmit({
      prompt: fullPrompt,
      duration_ms: duration * 1000,
      instrumental_only: instrumental,
      lyrics: lyrics || null
    });
  };

  const selectStyle = {
    ...styles.input,
    cursor: 'pointer',
    appearance: 'none',
    backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%23888' d='M6 8L1 3h10z'/%3E%3C/svg%3E")`,
    backgroundRepeat: 'no-repeat',
    backgroundPosition: 'right 12px center',
    paddingRight: '36px',
  };

  const hasStyle = mood || genre || theme || style || prompt;

  const stepBadge = (num) => (
    <span style={{
      background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)',
      color: '#fff',
      width: '24px',
      height: '24px',
      borderRadius: '50%',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      fontSize: '0.8rem',
      fontWeight: 'bold',
      flexShrink: 0,
    }}>{num}</span>
  );

  return (
    <form onSubmit={handleSubmit} style={styles.form}>
      {/* Collapsible Help */}
      <div style={{ marginBottom: '20px' }}>
        <button
          type="button"
          onClick={() => setShowHelp(!showHelp)}
          style={{
            background: 'none',
            border: 'none',
            color: '#60a5fa',
            cursor: 'pointer',
            fontSize: '0.9rem',
            padding: 0,
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
          }}
        >
          <span style={{ fontSize: '1.1rem' }}>ℹ️</span>
          How it works
          <span style={{ fontSize: '0.8rem' }}>{showHelp ? '▲' : '▼'}</span>
        </button>

        {showHelp && (
          <div style={{
            marginTop: '12px',
            padding: '15px',
            background: 'rgba(96, 165, 250, 0.1)',
            border: '1px solid rgba(96, 165, 250, 0.2)',
            borderRadius: '8px',
          }}>
            <ol style={{ margin: 0, paddingLeft: '20px', fontSize: '0.9rem', color: '#ccc', lineHeight: 1.8 }}>
              <li>Choose your song style using dropdowns</li>
              <li>Add your own description and lyrics</li>
              <li>AI generates a unique song for you!</li>
            </ol>
          </div>
        )}
      </div>

      {/* Step 1: Song Style */}
      <div style={styles.formGroup}>
        <label style={{ ...styles.label, display: 'flex', alignItems: 'center', gap: '8px' }}>
          {stepBadge(1)}
          Song Style
        </label>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '10px' }}>
          <select value={mood} onChange={(e) => setMood(e.target.value)} style={selectStyle}>
            {MUSIC_OPTIONS.moods.map(opt => (
              <option key={opt.value} value={opt.value}>{opt.label}</option>
            ))}
          </select>
          <select value={genre} onChange={(e) => setGenre(e.target.value)} style={selectStyle}>
            {MUSIC_OPTIONS.genres.map(opt => (
              <option key={opt.value} value={opt.value}>{opt.label}</option>
            ))}
          </select>
          <select value={theme} onChange={(e) => setTheme(e.target.value)} style={selectStyle}>
            {MUSIC_OPTIONS.themes.map(opt => (
              <option key={opt.value} value={opt.value}>{opt.label}</option>
            ))}
          </select>
          <select value={style} onChange={(e) => setStyle(e.target.value)} style={selectStyle}>
            {MUSIC_OPTIONS.styles.map(opt => (
              <option key={opt.value} value={opt.value}>{opt.label}</option>
            ))}
          </select>
        </div>

        {/* Selected Tags */}
        {(mood || genre || theme || style) && (
          <div style={{ marginTop: '10px', display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
            {mood && (
              <span style={{ padding: '4px 10px', background: 'rgba(99, 102, 241, 0.2)', borderRadius: '12px', fontSize: '0.8rem', color: '#a5b4fc' }}>
                {MUSIC_OPTIONS.moods.find(m => m.value === mood)?.label}
              </span>
            )}
            {genre && (
              <span style={{ padding: '4px 10px', background: 'rgba(16, 185, 129, 0.2)', borderRadius: '12px', fontSize: '0.8rem', color: '#6ee7b7' }}>
                {MUSIC_OPTIONS.genres.find(g => g.value === genre)?.label}
              </span>
            )}
            {theme && (
              <span style={{ padding: '4px 10px', background: 'rgba(245, 158, 11, 0.2)', borderRadius: '12px', fontSize: '0.8rem', color: '#fcd34d' }}>
                {MUSIC_OPTIONS.themes.find(t => t.value === theme)?.label}
              </span>
            )}
            {style && (
              <span style={{ padding: '4px 10px', background: 'rgba(236, 72, 153, 0.2)', borderRadius: '12px', fontSize: '0.8rem', color: '#f9a8d4' }}>
                {MUSIC_OPTIONS.styles.find(s => s.value === style)?.label}
              </span>
            )}
          </div>
        )}
      </div>

      {/* Step 2: Song Details */}
      <div style={styles.formGroup}>
        <label style={{ ...styles.label, display: 'flex', alignItems: 'center', gap: '8px' }}>
          {stepBadge(2)}
          Song Details
        </label>

        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Describe your song (e.g., 'An upbeat summer anthem with catchy hooks')..."
          style={{ ...styles.textarea, marginBottom: '12px' }}
          rows={2}
        />

        {!instrumental && (
          <textarea
            value={lyrics}
            onChange={(e) => {
              setLyrics(e.target.value);
              if (onLyricsChange) onLyricsChange(e.target.value);
            }}
            placeholder={generatedLyrics ? "Lyrics generated! Feel free to edit..." : "Enter lyrics or leave empty to auto-generate..."}
            style={{
              ...styles.textarea,
              ...(generatedLyrics && !lyrics ? { borderColor: '#4ade80' } : {})
            }}
            rows={4}
          />
        )}
      </div>

      {/* Advanced Options Toggle */}
      <div style={{ marginBottom: '20px' }}>
        <button
          type="button"
          onClick={() => setShowAdvanced(!showAdvanced)}
          style={{
            background: 'none',
            border: 'none',
            color: '#888',
            cursor: 'pointer',
            fontSize: '0.85rem',
            padding: 0,
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
          }}
        >
          <span>⚙️</span>
          Advanced Options
          <span style={{ fontSize: '0.8rem' }}>{showAdvanced ? '▲' : '▼'}</span>
        </button>

        {showAdvanced && (
          <div style={{
            marginTop: '15px',
            padding: '20px',
            background: 'rgba(255,255,255,0.03)',
            borderRadius: '12px',
            border: '1px solid rgba(255,255,255,0.1)',
          }}>
            {/* Duration */}
            <div style={{ marginBottom: '16px' }}>
              <label style={styles.label}>
                Duration: {duration}s ({Math.floor(duration / 60)}:{(duration % 60).toString().padStart(2, '0')})
              </label>
              <input
                type="range"
                min={10}
                max={300}
                value={duration}
                onChange={(e) => setDuration(Number(e.target.value))}
                style={styles.slider}
              />
            </div>

            {/* Instrumental Toggle */}
            <div>
              <label style={styles.checkboxLabel}>
                <input
                  type="checkbox"
                  checked={instrumental}
                  onChange={(e) => setInstrumental(e.target.checked)}
                  style={styles.checkbox}
                />
                Instrumental Only (no vocals)
              </label>
            </div>
          </div>
        )}
      </div>

      {/* Submit Button */}
      <button
        type="submit"
        style={{
          ...styles.submitButton,
          opacity: !hasStyle ? 0.5 : 1,
          padding: '16px 24px',
          fontSize: '1.1rem',
        }}
        disabled={loading || !hasStyle}
      >
        {loading ? '⏳ Generating...' : '🎵 Generate Music'}
      </button>
    </form>
  );
}

function VoiceCoverForm({ onSubmit, loading, customModels = [] }) {
  const [songFile, setSongFile] = useState(null);
  const [modelUrl, setModelUrl] = useState('');
  const [pitchShift, setPitchShift] = useState(0);
  const [showHelp, setShowHelp] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [useCustomModel, setUseCustomModel] = useState(false);
  const [selectedCustomModel, setSelectedCustomModel] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();

    const formData = new FormData();
    formData.append('song_file', songFile);
    formData.append('pitch_shift', pitchShift);

    if (useCustomModel && selectedCustomModel) {
      formData.append('voice_source', 'custom');
      formData.append('custom_model_id', selectedCustomModel);
    } else {
      formData.append('voice_source', 'url');
      formData.append('model_url', modelUrl);
    }

    onSubmit(formData);
  };

  const isSubmitDisabled = () => {
    if (loading || !songFile) return true;
    if (useCustomModel) {
      return !selectedCustomModel;
    }
    return !modelUrl;
  };

  return (
    <form onSubmit={handleSubmit} style={styles.form}>
      {/* Collapsible Help Section */}
      <div style={{ marginBottom: '20px' }}>
        <button
          type="button"
          onClick={() => setShowHelp(!showHelp)}
          style={{
            background: 'none',
            border: 'none',
            color: '#60a5fa',
            cursor: 'pointer',
            fontSize: '0.9rem',
            padding: 0,
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
          }}
        >
          <span style={{ fontSize: '1.1rem' }}>ℹ️</span>
          How it works
          <span style={{ fontSize: '0.8rem' }}>{showHelp ? '▲' : '▼'}</span>
        </button>

        {showHelp && (
          <div style={{
            marginTop: '12px',
            padding: '15px',
            background: 'rgba(96, 165, 250, 0.1)',
            border: '1px solid rgba(96, 165, 250, 0.2)',
            borderRadius: '8px',
          }}>
            <ol style={{ margin: 0, paddingLeft: '20px', fontSize: '0.9rem', color: '#ccc', lineHeight: 1.8 }}>
              <li>Upload the song you want to cover</li>
              <li>Upload your voice</li>
              <li>If you have a custom model, paste the model download link</li>
              <li>Generate!</li>
            </ol>
          </div>
        )}
      </div>

      {/* Step 1: Upload Song */}
      <div style={styles.formGroup}>
        <label style={{ ...styles.label, display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{
            background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)',
            color: '#fff',
            width: '24px',
            height: '24px',
            borderRadius: '50%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: '0.8rem',
            fontWeight: 'bold',
          }}>1</span>
          Upload Song
        </label>
        <div
          style={{
            border: songFile ? '2px solid #10b981' : '2px dashed rgba(255,255,255,0.2)',
            borderRadius: '12px',
            padding: '30px',
            textAlign: 'center',
            background: songFile ? 'rgba(16, 185, 129, 0.1)' : 'rgba(255,255,255,0.02)',
            cursor: 'pointer',
            transition: 'all 0.2s ease',
          }}
          onClick={() => document.getElementById('song-upload').click()}
        >
          <input
            id="song-upload"
            type="file"
            accept="audio/*"
            onChange={(e) => setSongFile(e.target.files[0])}
            style={{ display: 'none' }}
          />
          {songFile ? (
            <div>
              <span style={{ fontSize: '2rem' }}>✅</span>
              <p style={{ margin: '10px 0 0 0', color: '#10b981', fontWeight: '600' }}>
                {songFile.name}
              </p>
              <p style={{ margin: '5px 0 0 0', color: '#888', fontSize: '0.85rem' }}>
                Click to change
              </p>
            </div>
          ) : (
            <div>
              <span style={{ fontSize: '2rem' }}>🎵</span>
              <p style={{ margin: '10px 0 0 0', color: '#aaa' }}>
                Click to upload audio file
              </p>
              <p style={{ margin: '5px 0 0 0', color: '#666', fontSize: '0.85rem' }}>
                MP3, WAV, M4A supported
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Step 2: Voice Model */}
      <div style={styles.formGroup}>
        <label style={{ ...styles.label, display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{
            background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)',
            color: '#fff',
            width: '24px',
            height: '24px',
            borderRadius: '50%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: '0.8rem',
            fontWeight: 'bold',
          }}>2</span>
          Voice Model
        </label>

        {/* Voice Source Tabs */}
        <div style={{
          display: 'flex',
          gap: '8px',
          marginBottom: '12px',
        }}>
          <button
            type="button"
            onClick={() => {
              setUseCustomModel(false);
              setSelectedCustomModel('');
            }}
            style={{
              flex: 1,
              padding: '10px 16px',
              border: !useCustomModel ? '2px solid #10b981' : '1px solid rgba(255,255,255,0.2)',
              borderRadius: '8px',
              background: !useCustomModel ? 'rgba(16, 185, 129, 0.15)' : 'transparent',
              color: !useCustomModel ? '#10b981' : '#888',
              cursor: 'pointer',
              fontWeight: '600',
              fontSize: '0.9rem',
              transition: 'all 0.2s ease',
            }}
          >
            🔗 Paste URL
          </button>
          <button
            type="button"
            onClick={() => {
              setUseCustomModel(true);
              setModelUrl('');
            }}
            style={{
              flex: 1,
              padding: '10px 16px',
              border: useCustomModel ? '2px solid #ff6b35' : '1px solid rgba(255,255,255,0.2)',
              borderRadius: '8px',
              background: useCustomModel ? 'rgba(255, 107, 53, 0.15)' : 'transparent',
              color: useCustomModel ? '#ff6b35' : '#888',
              cursor: 'pointer',
              fontWeight: '600',
              fontSize: '0.9rem',
              transition: 'all 0.2s ease',
            }}
          >
            🧠 My Models
          </button>
        </div>

        {/* URL Input */}
        {!useCustomModel && (
          <input
            type="url"
            value={modelUrl}
            onChange={(e) => setModelUrl(e.target.value)}
            placeholder="Paste voice model URL"
            style={{
              ...styles.input,
              padding: '14px 16px',
              fontSize: '1rem',
            }}
          />
        )}

        {/* Saved Models Dropdown */}
        {useCustomModel && (
          <>
            {customModels.length === 0 ? (
              <div style={{
                padding: '20px',
                background: 'rgba(255,255,255,0.03)',
                borderRadius: '8px',
                border: '1px dashed rgba(255,255,255,0.2)',
                textAlign: 'center',
              }}>
                <p style={{ color: '#888', margin: 0, fontSize: '0.9rem' }}>
                  No saved models yet. Go to <strong>Voice Models</strong> tab to add one!
                </p>
              </div>
            ) : (
              <select
                value={selectedCustomModel}
                onChange={(e) => setSelectedCustomModel(e.target.value)}
                style={{
                  ...styles.select,
                  padding: '14px 16px',
                  fontSize: '1rem',
                }}
              >
                <option value="">Choose a saved model...</option>
                {customModels.map(model => (
                  <option key={model.id} value={model.id}>
                    {model.name}
                  </option>
                ))}
              </select>
            )}
          </>
        )}
      </div>

      {/* Advanced Options Toggle */}
      <div style={{ marginBottom: '20px' }}>
        <button
          type="button"
          onClick={() => setShowAdvanced(!showAdvanced)}
          style={{
            background: 'none',
            border: 'none',
            color: '#888',
            cursor: 'pointer',
            fontSize: '0.85rem',
            padding: 0,
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
          }}
        >
          <span>⚙️</span>
          Advanced Options
          <span style={{ fontSize: '0.8rem' }}>{showAdvanced ? '▲' : '▼'}</span>
        </button>

        {showAdvanced && (
          <div style={{
            marginTop: '15px',
            padding: '20px',
            background: 'rgba(255,255,255,0.03)',
            borderRadius: '12px',
            border: '1px solid rgba(255,255,255,0.1)',
          }}>
            <label style={styles.label}>
              Pitch Shift: {pitchShift > 0 ? '+' : ''}{pitchShift} semitones
            </label>
            <input
              type="range"
              min={-12}
              max={12}
              value={pitchShift}
              onChange={(e) => setPitchShift(Number(e.target.value))}
              style={styles.slider}
            />
            <p style={{ margin: '8px 0 0 0', color: '#666', fontSize: '0.8rem' }}>
              Adjust if the voice is too high or too low for the song
            </p>
          </div>
        )}
      </div>

      {/* Submit Button */}
      <button
        type="submit"
        style={{
          ...styles.submitButton,
          opacity: isSubmitDisabled() ? 0.5 : 1,
          padding: '16px 24px',
          fontSize: '1.1rem',
        }}
        disabled={isSubmitDisabled()}
      >
        {loading ? '⏳ Processing...' : '🎤 Create Voice Cover'}
      </button>
    </form>
  );
}

function CombinedForm({ onSubmit, loading, customModels = [], onGoToVoiceModels }) {
  const [prompt, setPrompt] = useState('');
  const [duration, setDuration] = useState(60);
  const [lyrics, setLyrics] = useState('');
  const [selectedModel, setSelectedModel] = useState('');
  const [conversionStrength, setConversionStrength] = useState(0.8);
  const [showHelp, setShowHelp] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Dropdown states
  const [mood, setMood] = useState('');
  const [genre, setGenre] = useState('');
  const [theme, setTheme] = useState('');
  const [style, setStyle] = useState('');

  // Build hidden prompt from selections
  const buildHiddenPrompt = () => {
    const parts = [];
    if (mood) {
      const moodLabel = MUSIC_OPTIONS.moods.find(m => m.value === mood)?.label.replace(/^[^\s]+\s/, '') || mood;
      parts.push(moodLabel.toLowerCase() + ' mood');
    }
    if (genre) {
      const genreLabel = MUSIC_OPTIONS.genres.find(g => g.value === genre)?.label.replace(/^[^\s]+\s/, '') || genre;
      parts.push(genreLabel + ' genre');
    }
    if (theme) {
      const themeLabel = MUSIC_OPTIONS.themes.find(t => t.value === theme)?.label.replace(/^[^\s]+\s/, '') || theme;
      parts.push(`theme of ${themeLabel.toLowerCase()}`);
    }
    if (style) {
      const styleLabel = MUSIC_OPTIONS.styles.find(s => s.value === style)?.label.replace(/^[^\s]+\s/, '') || style;
      parts.push(`${styleLabel.toLowerCase()} production style`);
    }
    return parts.length > 0 ? parts.join(', ') : '';
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    const hiddenPrompt = buildHiddenPrompt();
    const fullPrompt = [prompt, hiddenPrompt].filter(Boolean).join('. ');

    const formData = new FormData();
    formData.append('prompt', fullPrompt);
    formData.append('duration_ms', duration * 1000);
    formData.append('lyrics', lyrics || '');
    formData.append('custom_model_id', selectedModel);
    formData.append('conversion_strength', conversionStrength);

    onSubmit(formData);
  };

  const selectStyle = {
    ...styles.input,
    cursor: 'pointer',
    appearance: 'none',
    backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%23888' d='M6 8L1 3h10z'/%3E%3C/svg%3E")`,
    backgroundRepeat: 'no-repeat',
    backgroundPosition: 'right 12px center',
    paddingRight: '36px',
  };

  const hasModels = customModels && customModels.length > 0;
  const hasStyle = mood || genre || theme || style || prompt;

  const stepBadge = (num) => (
    <span style={{
      background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)',
      color: '#fff',
      width: '24px',
      height: '24px',
      borderRadius: '50%',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      fontSize: '0.8rem',
      fontWeight: 'bold',
      flexShrink: 0,
    }}>{num}</span>
  );

  return (
    <form onSubmit={handleSubmit} style={styles.form}>
      {/* Collapsible Help */}
      <div style={{ marginBottom: '20px' }}>
        <button
          type="button"
          onClick={() => setShowHelp(!showHelp)}
          style={{
            background: 'none',
            border: 'none',
            color: '#60a5fa',
            cursor: 'pointer',
            fontSize: '0.9rem',
            padding: 0,
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
          }}
        >
          <span style={{ fontSize: '1.1rem' }}>ℹ️</span>
          How it works
          <span style={{ fontSize: '0.8rem' }}>{showHelp ? '▲' : '▼'}</span>
        </button>

        {showHelp && (
          <div style={{
            marginTop: '12px',
            padding: '15px',
            background: 'rgba(96, 165, 250, 0.1)',
            border: '1px solid rgba(96, 165, 250, 0.2)',
            borderRadius: '8px',
          }}>
            <ol style={{ margin: 0, paddingLeft: '20px', fontSize: '0.9rem', color: '#ccc', lineHeight: 1.8 }}>
              <li>Choose your song style (mood, genre, etc.)</li>
              <li>Add customized details and lyrics</li>
              <li>Select your saved voice model</li>
              <li>AI generates original music sung in your voice!</li>
            </ol>
          </div>
        )}
      </div>

      {/* Step 1: Song Style */}
      <div style={styles.formGroup}>
        <label style={{ ...styles.label, display: 'flex', alignItems: 'center', gap: '8px' }}>
          {stepBadge(1)}
          Song Style
        </label>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '10px' }}>
          <select value={mood} onChange={(e) => setMood(e.target.value)} style={selectStyle}>
            {MUSIC_OPTIONS.moods.map(opt => (
              <option key={opt.value} value={opt.value}>{opt.label}</option>
            ))}
          </select>
          <select value={genre} onChange={(e) => setGenre(e.target.value)} style={selectStyle}>
            {MUSIC_OPTIONS.genres.map(opt => (
              <option key={opt.value} value={opt.value}>{opt.label}</option>
            ))}
          </select>
          <select value={theme} onChange={(e) => setTheme(e.target.value)} style={selectStyle}>
            {MUSIC_OPTIONS.themes.map(opt => (
              <option key={opt.value} value={opt.value}>{opt.label}</option>
            ))}
          </select>
          <select value={style} onChange={(e) => setStyle(e.target.value)} style={selectStyle}>
            {MUSIC_OPTIONS.styles.map(opt => (
              <option key={opt.value} value={opt.value}>{opt.label}</option>
            ))}
          </select>
        </div>

        {/* Selected Tags */}
        {hasStyle && (
          <div style={{ marginTop: '10px', display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
            {mood && (
              <span style={{ padding: '4px 10px', background: 'rgba(99, 102, 241, 0.2)', borderRadius: '12px', fontSize: '0.8rem', color: '#a5b4fc' }}>
                {MUSIC_OPTIONS.moods.find(m => m.value === mood)?.label}
              </span>
            )}
            {genre && (
              <span style={{ padding: '4px 10px', background: 'rgba(16, 185, 129, 0.2)', borderRadius: '12px', fontSize: '0.8rem', color: '#6ee7b7' }}>
                {MUSIC_OPTIONS.genres.find(g => g.value === genre)?.label}
              </span>
            )}
            {theme && (
              <span style={{ padding: '4px 10px', background: 'rgba(245, 158, 11, 0.2)', borderRadius: '12px', fontSize: '0.8rem', color: '#fcd34d' }}>
                {MUSIC_OPTIONS.themes.find(t => t.value === theme)?.label}
              </span>
            )}
            {style && (
              <span style={{ padding: '4px 10px', background: 'rgba(236, 72, 153, 0.2)', borderRadius: '12px', fontSize: '0.8rem', color: '#f9a8d4' }}>
                {MUSIC_OPTIONS.styles.find(s => s.value === style)?.label}
              </span>
            )}
          </div>
        )}
      </div>

      {/* Step 2: Details & Lyrics */}
      <div style={styles.formGroup}>
        <label style={{ ...styles.label, display: 'flex', alignItems: 'center', gap: '8px' }}>
          {stepBadge(2)}
          Details & Lyrics
        </label>

        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Describe your song (e.g., 'An upbeat pop song about summer road trips')..."
          style={{ ...styles.textarea, marginBottom: '12px' }}
          rows={2}
        />

        <textarea
          value={lyrics}
          onChange={(e) => setLyrics(e.target.value)}
          placeholder="Enter custom lyrics (optional)..."
          style={styles.textarea}
          rows={3}
        />
      </div>

      {/* Step 3: Voice Model */}
      <div style={styles.formGroup}>
        <label style={{ ...styles.label, display: 'flex', alignItems: 'center', gap: '8px' }}>
          {stepBadge(3)}
          Your Voice
        </label>
        {hasModels ? (
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            style={{ ...selectStyle, padding: '14px 16px', fontSize: '1rem' }}
          >
            <option value="">Choose a voice model...</option>
            {customModels.map(model => (
              <option key={model.id} value={model.id}>
                {model.name}
              </option>
            ))}
          </select>
        ) : (
          <div style={{
            padding: '20px',
            background: 'rgba(245, 158, 11, 0.1)',
            border: '1px solid rgba(245, 158, 11, 0.3)',
            borderRadius: '12px',
            textAlign: 'center',
          }}>
            <p style={{ margin: '0 0 12px 0', color: '#fcd34d', fontSize: '0.95rem' }}>
              ⚠️ No voice models saved yet
            </p>
            <button
              type="button"
              onClick={onGoToVoiceModels}
              style={{
                padding: '10px 20px',
                background: 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)',
                border: 'none',
                borderRadius: '8px',
                color: '#fff',
                fontWeight: '600',
                cursor: 'pointer',
              }}
            >
              Add Voice Model →
            </button>
          </div>
        )}
      </div>

      {/* Advanced Options Toggle */}
      <div style={{ marginBottom: '20px' }}>
        <button
          type="button"
          onClick={() => setShowAdvanced(!showAdvanced)}
          style={{
            background: 'none',
            border: 'none',
            color: '#888',
            cursor: 'pointer',
            fontSize: '0.85rem',
            padding: 0,
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
          }}
        >
          <span>⚙️</span>
          Advanced Options
          <span style={{ fontSize: '0.8rem' }}>{showAdvanced ? '▲' : '▼'}</span>
        </button>

        {showAdvanced && (
          <div style={{
            marginTop: '15px',
            padding: '20px',
            background: 'rgba(255,255,255,0.03)',
            borderRadius: '12px',
            border: '1px solid rgba(255,255,255,0.1)',
          }}>
            {/* Duration */}
            <div style={{ marginBottom: '16px' }}>
              <label style={styles.label}>
                Duration: {duration}s ({Math.floor(duration / 60)}:{(duration % 60).toString().padStart(2, '0')})
              </label>
              <input
                type="range"
                min={10}
                max={180}
                value={duration}
                onChange={(e) => setDuration(Number(e.target.value))}
                style={styles.slider}
              />
            </div>

            {/* Voice Strength */}
            <div>
              <label style={styles.label}>
                Voice Strength: {(conversionStrength * 100).toFixed(0)}%
              </label>
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={conversionStrength}
                onChange={(e) => setConversionStrength(Number(e.target.value))}
                style={styles.slider}
              />
            </div>
          </div>
        )}
      </div>

      {/* Submit Button */}
      <button
        type="submit"
        style={{
          ...styles.submitButton,
          opacity: (!hasModels || !selectedModel || !hasStyle) ? 0.5 : 1,
          padding: '16px 24px',
          fontSize: '1.1rem',
        }}
        disabled={loading || !hasModels || !selectedModel || !hasStyle}
      >
        {loading ? '⏳ Generating...' : '✨ Generate with Your Voice'}
      </button>
    </form>
  );
}

function TrainModelForm({ onTrainSubmit, onUploadSubmit, loading, customModels = [], onDeleteModel, trainingStatus }) {
  const [mode, setMode] = useState('train');
  const [modelName, setModelName] = useState('');
  const [showHelp, setShowHelp] = useState(false);
  const auth = useAuth();

  // Voice training state
  const [voiceSamples, setVoiceSamples] = useState([]);

  // Upload mode state - URL method
  const [modelUrl, setModelUrl] = useState('');

  const handleVoiceSamplesChange = (e) => {
    setVoiceSamples(Array.from(e.target.files));
  };

  const handleTrainSubmit = (e) => {
    e.preventDefault();
    const formData = new FormData();
    formData.append('model_name', modelName);
    voiceSamples.forEach((file, index) => {
      formData.append(`voice_sample_${index}`, file);
    });
    formData.append('sample_count', voiceSamples.length);
    onTrainSubmit(formData);
  };

  const handleUploadSubmit = (e) => {
    e.preventDefault();
    onUploadSubmit({ model_name: modelName, model_url: modelUrl });
  };

  const getTotalDuration = () => {
    const totalSize = voiceSamples.reduce((sum, file) => sum + file.size, 0);
    const estimatedMinutes = Math.round(totalSize / (1024 * 1024) * 0.5);
    return estimatedMinutes;
  };

  // Show sign-in prompt if not authenticated
  if (!auth?.isAuthenticated) {
    return (
      <div style={{
        padding: '40px',
        textAlign: 'center',
        background: 'rgba(255,255,255,0.03)',
        borderRadius: '12px',
        border: '1px solid rgba(255,255,255,0.1)',
      }}>
        <div style={{ fontSize: '48px', marginBottom: '20px' }}>🔐</div>
        <h3 style={{ color: '#fff', marginBottom: '12px' }}>Sign In Required</h3>
        <p style={{ color: '#888', marginBottom: '20px' }}>
          Please sign in to train and manage your voice models.
          <br />
          Your models are private and only visible to you.
        </p>
        <p style={{ color: '#666', fontSize: '0.9rem' }}>
          Click the <strong style={{ color: '#818cf8' }}>Sign In</strong> button in the top right corner.
        </p>
      </div>
    );
  }

  return (
    <div>
      {/* Collapsible Help */}
      <div style={{ marginBottom: '20px' }}>
        <button
          type="button"
          onClick={() => setShowHelp(!showHelp)}
          style={{
            background: 'none',
            border: 'none',
            color: '#60a5fa',
            cursor: 'pointer',
            fontSize: '0.9rem',
            padding: 0,
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
          }}
        >
          <span style={{ fontSize: '1.1rem' }}>ℹ️</span>
          How it works
          <span style={{ fontSize: '0.8rem' }}>{showHelp ? '▲' : '▼'}</span>
        </button>

        {showHelp && (
          <div style={{
            marginTop: '12px',
            padding: '15px',
            background: 'rgba(96, 165, 250, 0.1)',
            border: '1px solid rgba(96, 165, 250, 0.2)',
            borderRadius: '8px',
          }}>
            <p style={{ margin: '0 0 10px 0', fontSize: '0.9rem', color: '#ccc' }}>
              <strong>🎤 Train My Voice:</strong> Upload recordings of your voice and AI creates a custom model.
            </p>
            <p style={{ margin: 0, fontSize: '0.9rem', color: '#ccc' }}>
              <strong>🔗 Add from URL:</strong> Recommended to use pre-trained models from voice-models.com (celebrities, characters, etc.)
            </p>
          </div>
        )}
      </div>

      {/* Tab-style Mode Selection */}
      <div style={{
        display: 'flex',
        gap: '8px',
        marginBottom: '24px',
      }}>
        <button
          type="button"
          onClick={() => {
            setMode('train');
            setModelName('');
            setModelUrl('');
          }}
          style={{
            flex: 1,
            padding: '14px 16px',
            border: mode === 'train' ? '2px solid #10b981' : '1px solid rgba(255,255,255,0.2)',
            borderRadius: '10px',
            background: mode === 'train' ? 'rgba(16, 185, 129, 0.15)' : 'transparent',
            color: mode === 'train' ? '#10b981' : '#888',
            cursor: 'pointer',
            fontWeight: '600',
            fontSize: '0.95rem',
            transition: 'all 0.2s ease',
          }}
        >
          🎤 Train My Voice
        </button>
        <button
          type="button"
          onClick={() => {
            setMode('url');
            setModelName('');
            setVoiceSamples([]);
          }}
          style={{
            flex: 1,
            padding: '14px 16px',
            border: mode === 'url' ? '2px solid #6366f1' : '1px solid rgba(255,255,255,0.2)',
            borderRadius: '10px',
            background: mode === 'url' ? 'rgba(99, 102, 241, 0.15)' : 'transparent',
            color: mode === 'url' ? '#6366f1' : '#888',
            cursor: 'pointer',
            fontWeight: '600',
            fontSize: '0.95rem',
            transition: 'all 0.2s ease',
          }}
        >
          🔗 Add from URL
        </button>
      </div>

      {/* Train Voice Form */}
      {mode === 'train' && (
        <form onSubmit={handleTrainSubmit} style={styles.form}>
          <div style={styles.formGroup}>
            <label style={styles.label}>Voice Name *</label>
            <input
              type="text"
              value={modelName}
              onChange={(e) => setModelName(e.target.value)}
              placeholder="e.g., My Voice, John's Voice..."
              style={styles.input}
              required
            />
          </div>

          <div style={styles.formGroup}>
            <label style={styles.label}>Voice Recordings *</label>
            <div style={{
              padding: '30px 20px',
              background: 'rgba(255,255,255,0.03)',
              borderRadius: '12px',
              border: '2px dashed rgba(255,255,255,0.2)',
              textAlign: 'center',
              cursor: 'pointer',
            }}>
              <input
                type="file"
                accept="audio/*"
                multiple
                onChange={handleVoiceSamplesChange}
                style={{ display: 'none' }}
                id="voice-upload"
                required
              />
              <label htmlFor="voice-upload" style={{ cursor: 'pointer' }}>
                <div style={{ fontSize: '2rem', marginBottom: '10px' }}>🎙️</div>
                <p style={{ color: '#888', margin: 0 }}>
                  Click to upload audio files<br />
                  <span style={{ fontSize: '0.85rem' }}>3-10 minutes total recommended</span>
                </p>
              </label>
            </div>

            {voiceSamples.length > 0 && (
              <div style={{ marginTop: '12px', padding: '12px', background: 'rgba(16,185,129,0.1)', borderRadius: '8px' }}>
                <p style={{ margin: 0, color: '#10b981', fontWeight: '500' }}>
                  ✅ {voiceSamples.length} file(s) selected (~{getTotalDuration()} min)
                </p>
                <div style={{ maxHeight: '80px', overflowY: 'auto', fontSize: '0.85rem', color: '#888', marginTop: '8px' }}>
                  {voiceSamples.map((file, i) => (
                    <div key={i}>{file.name}</div>
                  ))}
                </div>
              </div>
            )}

            <p style={{ ...styles.helpText, marginTop: '10px' }}>
              💡 Tips: Clear voice, quiet room, no background music
            </p>
          </div>

          {trainingStatus && (
            <div style={{
              padding: '15px',
              background: trainingStatus.status === 'failed' ? 'rgba(239, 68, 68, 0.1)' : 'rgba(16, 185, 129, 0.1)',
              border: `1px solid ${trainingStatus.status === 'failed' ? 'rgba(239, 68, 68, 0.3)' : 'rgba(16, 185, 129, 0.3)'}`,
              borderRadius: '8px',
              marginBottom: '15px'
            }}>
              <p style={{ margin: 0, color: trainingStatus.status === 'failed' ? '#ef4444' : '#10b981' }}>
                {trainingStatus.status === 'processing' && '⏳ '}
                {trainingStatus.status === 'completed' && '✅ '}
                {trainingStatus.status === 'failed' && '❌ '}
                {trainingStatus.message}
              </p>
              {trainingStatus.progress > 0 && trainingStatus.status === 'processing' && (
                <div style={{ marginTop: '10px' }}>
                  <div style={{
                    height: '8px',
                    background: 'rgba(255,255,255,0.1)',
                    borderRadius: '4px',
                    overflow: 'hidden'
                  }}>
                    <div style={{
                      height: '100%',
                      width: `${trainingStatus.progress}%`,
                      background: '#10b981',
                      transition: 'width 0.3s'
                    }} />
                  </div>
                  <p style={{ margin: '5px 0 0 0', fontSize: '0.85rem', color: '#888' }}>
                    {trainingStatus.progress}% complete
                  </p>
                </div>
              )}
            </div>
          )}

          <button
            type="submit"
            style={{
              ...styles.submitButton,
              background: 'linear-gradient(135deg, #10b981, #059669)',
              opacity: (!modelName || voiceSamples.length === 0) ? 0.5 : 1,
            }}
            disabled={loading || !modelName || voiceSamples.length === 0}
          >
            {loading ? '⏳ Training (~10-20 min)...' : '🚀 Start Training'}
          </button>
        </form>
      )}

      {/* Add from URL Form */}
      {mode === 'url' && (
        <form onSubmit={handleUploadSubmit} style={styles.form}>
          <div style={{
            padding: '15px',
            background: 'rgba(99, 102, 241, 0.1)',
            border: '1px solid rgba(99, 102, 241, 0.2)',
            borderRadius: '8px',
            marginBottom: '20px'
          }}>
            <p style={{ margin: 0, fontSize: '0.9rem', color: '#ccc' }}>
              Find voice models at <a href="https://voice-models.com" target="_blank" rel="noopener noreferrer" style={{ color: '#818cf8' }}>voice-models.com</a> – celebrities, characters, singers & more!
            </p>
          </div>

          <div style={styles.formGroup}>
            <label style={styles.label}>Model Name *</label>
            <input
              type="text"
              value={modelName}
              onChange={(e) => setModelName(e.target.value)}
              placeholder="e.g., Taylor Swift, Morgan Freeman..."
              style={styles.input}
              required
            />
          </div>

          <div style={styles.formGroup}>
            <label style={styles.label}>Model URL *</label>
            <input
              type="url"
              value={modelUrl}
              onChange={(e) => setModelUrl(e.target.value)}
              placeholder="https://huggingface.co/.../model.zip"
              style={styles.input}
              required
            />
            <p style={styles.helpText}>
              Paste the direct download URL to an RVC model (.zip or .pth)
            </p>
          </div>

          <button
            type="submit"
            style={{
              ...styles.submitButton,
              background: 'linear-gradient(135deg, #6366f1, #4f46e5)',
              opacity: (!modelName || !modelUrl) ? 0.5 : 1,
            }}
            disabled={loading || !modelName || !modelUrl}
          >
            {loading ? '⏳ Saving...' : '✅ Add Voice Model'}
          </button>
        </form>
      )}

      {/* Models Lists */}
      <div style={{ marginTop: '40px' }}>
        {/* User's Custom Models */}
        <h3 style={{ color: '#fff', marginBottom: '20px', display: 'flex', alignItems: 'center', gap: '10px' }}>
          📂 My Custom Models
          <span style={{ fontSize: '0.9rem', color: '#888', fontWeight: 'normal' }}>
            ({customModels.filter(m => !m.is_default).length})
          </span>
        </h3>

        {customModels.filter(m => !m.is_default).length === 0 ? (
          <div style={{
            padding: '30px 20px',
            background: 'rgba(255,255,255,0.03)',
            borderRadius: '12px',
            border: '1px dashed rgba(255,255,255,0.2)',
            textAlign: 'center',
            marginBottom: '30px'
          }}>
            <p style={{ color: '#888', margin: 0 }}>
              No custom models yet. Train your voice or add from URL above!
            </p>
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', marginBottom: '30px' }}>
            {customModels.filter(m => !m.is_default).map(model => (
              <div
                key={model.id}
                style={{
                  padding: '15px 20px',
                  background: 'rgba(255,255,255,0.05)',
                  borderRadius: '10px',
                  border: '1px solid rgba(255,255,255,0.1)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between'
                }}
              >
                <div>
                  <div style={{ color: '#fff', fontWeight: '500', fontSize: '1.05rem' }}>
                    {model.type === 'trained' ? '🎤' : '🔗'} {model.name}
                  </div>
                  <div style={{ color: '#888', fontSize: '0.85rem', marginTop: '4px' }}>
                    {model.type === 'trained' ? 'Trained' : 'From URL'} • {new Date(model.created_at).toLocaleDateString()}
                  </div>
                </div>
                <button
                  onClick={() => onDeleteModel(model.id)}
                  style={{
                    background: 'rgba(239, 68, 68, 0.2)',
                    border: '1px solid rgba(239, 68, 68, 0.3)',
                    color: '#ef4444',
                    padding: '8px 16px',
                    borderRadius: '6px',
                    cursor: 'pointer',
                    fontSize: '0.9rem'
                  }}
                >
                  🗑️ Delete
                </button>
              </div>
            ))}
          </div>
        )}

        {/* Default Models */}
        <h3 style={{ color: '#fff', marginBottom: '20px', display: 'flex', alignItems: 'center', gap: '10px' }}>
          ⭐ Celebrity Voices
          <span style={{ fontSize: '0.9rem', color: '#888', fontWeight: 'normal' }}>
            (Available to everyone)
          </span>
        </h3>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: '12px' }}>
          {customModels.filter(m => m.is_default).map(model => (
            <div
              key={model.id}
              style={{
                padding: '20px',
                background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%)',
                borderRadius: '12px',
                border: '1px solid rgba(99, 102, 241, 0.3)',
                textAlign: 'center',
              }}
            >
              <div style={{ fontSize: '2rem', marginBottom: '8px' }}>⭐</div>
              <div style={{ color: '#fff', fontWeight: '600', fontSize: '1rem' }}>
                {model.name}
              </div>
              <div style={{ color: '#a5b4fc', fontSize: '0.8rem', marginTop: '4px' }}>
                Celebrity Voice
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function AnimatedProgress({ isActive, isComplete, isFailed, outputUrl, lyrics, error }) {
  const progress = useAnimatedProgress(isActive && !isComplete && !isFailed, isComplete ? 100 : null);

  const displayProgress = isComplete ? 100 : isFailed ? progress : progress;

  const getStatusColor = () => {
    if (isComplete) return '#4ade80';
    if (isFailed) return '#f87171';
    return '#60a5fa';
  };

  const getStatusText = () => {
    if (isComplete) return '✅ Complete!';
    if (isFailed) return '❌ Failed';
    if (progress < 30) return '🎵 Initializing...';
    if (progress < 60) return '🎼 Composing music...';
    if (progress < 85) return '🎤 Generating vocals...';
    return '✨ Finalizing...';
  };

  if (!isActive && !isComplete && !isFailed) return null;

  return (
    <div style={styles.progressContainer}>
      <div style={styles.progressHeader}>
        <span style={styles.progressTitle}>{getStatusText()}</span>
        <span style={{ ...styles.progressStatus, color: getStatusColor() }}>
          {displayProgress}%
        </span>
      </div>

      <div style={styles.progressBar}>
        <div
          style={{
            ...styles.progressFill,
            width: `${displayProgress}%`,
            backgroundColor: getStatusColor(),
            transition: 'width 0.3s ease-out'
          }}
        />
      </div>

      {isFailed && error && (
        <p style={styles.errorText}>{error}</p>
      )}

      {isComplete && outputUrl && (
        <div style={styles.resultContainer}>
          <audio
            controls
            src={outputUrl}
            style={styles.audioPlayer}
          />

          {lyrics && (
            <div style={styles.lyricsDisplay}>
              <h4 style={{ marginBottom: '10px', color: '#ccc' }}>📝 Generated Lyrics:</h4>
              <pre style={styles.lyricsText}>{lyrics}</pre>
            </div>
          )}

          <a
            href={outputUrl}
            download
            style={styles.downloadButton}
          >
            ⬇️ Download
          </a>
        </div>
      )}
    </div>
  );
}

function GenerationHistory() {
  const [jobs, setJobs] = useState([]);
  const [backendAvailable, setBackendAvailable] = useState(true);
  const { fetchApi } = useApi();

  useEffect(() => {
    let mounted = true;

    const loadJobs = async () => {
      try {
        const data = await fetchApi('/api/jobs');
        if (mounted) {
          setJobs(data.jobs || []);
          setBackendAvailable(true);
        }
      } catch (error) {
        if (mounted) {
          setBackendAvailable(false);
        }
      }
    };

    loadJobs();
    const interval = setInterval(loadJobs, 10000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, [fetchApi]);



  if (jobs.length === 0) return null;

  const getFlowLabel = (flow) => {
    switch (flow) {
      case 'original_music': return '🎼 Original';
      case 'voice_cover': return '🎤 Cover';
      case 'combined': return '✨ Combined';
      default: return flow;
    }
  };

  const getStatusLabel = (status) => {
    switch (status) {
      case 'completed': return '✅ Done';
      case 'failed': return '❌ Failed';
      case 'processing': return '⏳ Processing';
      case 'pending': return '🕐 Pending';
      default: return status;
    }
  };

  return (
    <div style={styles.historyContainer}>
      <h3 style={styles.historyTitle}>🎵 Generation History</h3>
      <div style={styles.jobsList}>
        {jobs.slice(0, 5).map(job => (
          <div key={job.job_id} style={styles.jobItem}>
            <div style={styles.jobInfo}>
              <span style={styles.jobFlow}>{getFlowLabel(job.flow)}</span>
              <span style={{
                ...styles.jobStatus,
                backgroundColor: job.status === 'completed' ? 'rgba(74,222,128,0.2)' :
                  job.status === 'failed' ? 'rgba(248,113,113,0.2)' :
                    'rgba(255,255,255,0.1)'
              }}>
                {getStatusLabel(job.status)}
              </span>
            </div>
            {job.output_url && job.status === 'completed' && (
              <audio
                controls
                src={job.output_url}
                style={styles.miniPlayer}
              />
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

// ============================================================================
// Main App
// ============================================================================

// ============================================================================
// Song History Sidebar
// ============================================================================

function SongHistorySidebar({ isOpen, onClose, songs, onUpdateSong, onDeleteSong }) {
  const [editingId, setEditingId] = useState(null);
  const [editName, setEditName] = useState('');

  const handleStartEdit = (song) => {
    setEditingId(song.id);
    setEditName(song.name || `Song ${song.id.slice(0, 8)}`);
  };

  const handleSaveEdit = (songId) => {
    onUpdateSong(songId, editName);
    setEditingId(null);
    setEditName('');
  };

  const handleCancelEdit = () => {
    setEditingId(null);
    setEditName('');
  };

  const formatDate = (dateStr) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <>
      {/* Overlay */}
      {isOpen && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'rgba(0,0,0,0.5)',
            zIndex: 998,
          }}
          onClick={onClose}
        />
      )}

      {/* Sidebar */}
      <div style={{
        position: 'fixed',
        top: 0,
        left: isOpen ? 0 : '-320px',
        width: '320px',
        height: '100vh',
        background: '#1a1a2e',
        borderRight: '1px solid rgba(255,255,255,0.1)',
        zIndex: 999,
        transition: 'left 0.3s ease',
        display: 'flex',
        flexDirection: 'column',
        boxShadow: isOpen ? '4px 0 20px rgba(0,0,0,0.3)' : 'none',
      }}>
        {/* Header */}
        <div style={{
          padding: '20px',
          borderBottom: '1px solid rgba(255,255,255,0.1)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}>
          <h3 style={{ margin: 0, color: '#fff', fontSize: '1.1rem' }}>
            🎵 My Songs
          </h3>
          <button
            onClick={onClose}
            style={{
              background: 'none',
              border: 'none',
              color: '#888',
              fontSize: '24px',
              cursor: 'pointer',
              padding: '4px',
            }}
          >
            ×
          </button>
        </div>

        {/* Songs List */}
        <div style={{
          flex: 1,
          overflowY: 'auto',
          padding: '10px',
        }}>
          {songs.length === 0 ? (
            <div style={{
              padding: '40px 20px',
              textAlign: 'center',
              color: '#666',
            }}>
              <div style={{ fontSize: '48px', marginBottom: '10px' }}>🎶</div>
              <p>No songs yet</p>
              <p style={{ fontSize: '0.85rem' }}>Generate your first song to see it here!</p>
            </div>
          ) : (
            songs.map((song) => (
              <div
                key={song.id}
                style={{
                  padding: '12px',
                  background: 'rgba(255,255,255,0.03)',
                  borderRadius: '8px',
                  marginBottom: '10px',
                  border: '1px solid rgba(255,255,255,0.05)',
                }}
              >
                {editingId === song.id ? (
                  // Edit mode
                  <div>
                    <input
                      type="text"
                      value={editName}
                      onChange={(e) => setEditName(e.target.value)}
                      style={{
                        width: '100%',
                        padding: '8px',
                        background: 'rgba(255,255,255,0.1)',
                        border: '1px solid rgba(255,255,255,0.2)',
                        borderRadius: '4px',
                        color: '#fff',
                        marginBottom: '8px',
                      }}
                      autoFocus
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') handleSaveEdit(song.id);
                        if (e.key === 'Escape') handleCancelEdit();
                      }}
                    />
                    <div style={{ display: 'flex', gap: '8px' }}>
                      <button
                        onClick={() => handleSaveEdit(song.id)}
                        style={{
                          flex: 1,
                          padding: '6px',
                          background: '#10b981',
                          border: 'none',
                          borderRadius: '4px',
                          color: '#fff',
                          cursor: 'pointer',
                          fontSize: '0.85rem',
                        }}
                      >
                        ✓ Save
                      </button>
                      <button
                        onClick={handleCancelEdit}
                        style={{
                          flex: 1,
                          padding: '6px',
                          background: 'rgba(255,255,255,0.1)',
                          border: 'none',
                          borderRadius: '4px',
                          color: '#888',
                          cursor: 'pointer',
                          fontSize: '0.85rem',
                        }}
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                ) : (
                  // View mode
                  <>
                    <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: '8px' }}>
                      <div style={{ flex: 1 }}>
                        <div style={{ color: '#fff', fontWeight: '500', marginBottom: '4px' }}>
                          {song.name || `Song ${song.id.slice(0, 8)}`}
                        </div>
                        <div style={{ fontSize: '0.75rem', color: '#666' }}>
                          {formatDate(song.created_at)}
                        </div>
                      </div>
                      <div style={{ display: 'flex', gap: '4px' }}>
                        <button
                          onClick={() => handleStartEdit(song)}
                          style={{
                            background: 'none',
                            border: 'none',
                            color: '#888',
                            cursor: 'pointer',
                            padding: '4px',
                            fontSize: '14px',
                          }}
                          title="Edit name"
                        >
                          ✏️
                        </button>
                        <button
                          onClick={() => onDeleteSong(song.id)}
                          style={{
                            background: 'none',
                            border: 'none',
                            color: '#888',
                            cursor: 'pointer',
                            padding: '4px',
                            fontSize: '14px',
                          }}
                          title="Delete"
                        >
                          🗑️
                        </button>
                      </div>
                    </div>

                    {song.type && (
                      <div style={{
                        fontSize: '0.7rem',
                        color: '#888',
                        marginBottom: '8px',
                        display: 'inline-block',
                        padding: '2px 6px',
                        background: 'rgba(255,255,255,0.05)',
                        borderRadius: '4px',
                      }}>
                        {song.type === 'music' ? '🎵 Generated' : song.type === 'cover' ? '🎤 Cover' : '🎭 Combined'}
                      </div>
                    )}

                    {/* Mini audio player */}
                    <audio
                      src={song.output_url}
                      controls
                      style={{
                        width: '100%',
                        height: '32px',
                        marginBottom: '8px',
                      }}
                    />

                    {/* Download button */}
                    <a
                      href={song.output_url}
                      download={`${song.name || 'song'}.mp3`}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        gap: '6px',
                        padding: '8px',
                        background: 'rgba(99, 102, 241, 0.2)',
                        border: '1px solid rgba(99, 102, 241, 0.3)',
                        borderRadius: '6px',
                        color: '#818cf8',
                        textDecoration: 'none',
                        fontSize: '0.85rem',
                      }}
                    >
                      ⬇️ Download
                    </a>
                  </>
                )}
              </div>
            ))
          )}
        </div>
      </div>
    </>
  );
}

// Hamburger menu button
function HamburgerButton({ onClick, songCount }) {
  return (
    <button
      onClick={onClick}
      style={{
        position: 'fixed',
        top: '20px',
        left: '20px',
        zIndex: 997,
        background: 'rgba(30, 30, 50, 0.9)',
        border: '1px solid rgba(255,255,255,0.1)',
        borderRadius: '10px',
        padding: '12px',
        cursor: 'pointer',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: '4px',
        backdropFilter: 'blur(10px)',
      }}
      title="My Songs"
    >
      <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
        <div style={{ width: '20px', height: '2px', background: '#fff', borderRadius: '1px' }} />
        <div style={{ width: '20px', height: '2px', background: '#fff', borderRadius: '1px' }} />
        <div style={{ width: '20px', height: '2px', background: '#fff', borderRadius: '1px' }} />
      </div>
      {songCount > 0 && (
        <span style={{
          fontSize: '0.7rem',
          color: '#10b981',
          fontWeight: 'bold',
        }}>
          {songCount}
        </span>
      )}
    </button>
  );
}

function AppContent() {
  const location = useLocation();
  const navigate = useNavigate();

  // Derive activeTab from current URL path
  const getActiveTab = () => {
    const path = location.pathname;
    if (path === '/cover') return 'cover';
    if (path === '/combined') return 'combined';
    if (path === '/voice-models') return 'train';
    if (path === '/login' || path === '/signup') return 'music'; // Default for auth pages
    return 'music'; // Default
  };
  const activeTab = getActiveTab();
  const [loading, setLoading] = useState(false);
  const [currentJobId, setCurrentJobId] = useState(null);
  const [error, setError] = useState(null);
  const [jobComplete, setJobComplete] = useState(false);
  const [jobFailed, setJobFailed] = useState(false);
  const [outputUrl, setOutputUrl] = useState(null);
  const [generatedLyrics, setGeneratedLyrics] = useState('');
  const [customModels, setCustomModels] = useState([]);
  const { fetchApi } = useApi();
  const auth = useAuth();

  // Auth modal state for /login and /signup routes
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [authMode, setAuthMode] = useState('login');

  // Handle /login and /signup routes
  useEffect(() => {
    if (location.pathname === '/login') {
      setAuthMode('login');
      setShowAuthModal(true);
    } else if (location.pathname === '/signup') {
      setAuthMode('signup');
      setShowAuthModal(true);
    } else {
      setShowAuthModal(false);
    }
  }, [location.pathname]);

  // Handle auth modal close - navigate back to previous page
  const handleAuthModalClose = () => {
    setShowAuthModal(false);
    navigate('/music');
  };

  // Song history state
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [songHistory, setSongHistory] = useState(() => {
    try {
      const saved = localStorage.getItem('songHistory');
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });

  // Save song history to localStorage
  useEffect(() => {
    try {
      localStorage.setItem('songHistory', JSON.stringify(songHistory));
    } catch (e) {
      console.error('Failed to save song history:', e);
    }
  }, [songHistory]);

  // Add song to history (with duplicate check)
  const addSongToHistory = (song) => {
    setSongHistory(prev => {
      // Check if song already exists
      const exists = prev.some(s => s.id === song.job_id || s.id === song.id);
      if (exists) {
        return prev; // Don't add duplicate
      }

      return [{
        id: song.job_id || song.id || Date.now().toString(),
        name: song.name || null,
        output_url: song.output_url,
        type: song.type || 'music',
        created_at: new Date().toISOString(),
        prompt: song.prompt || null,
      }, ...prev].slice(0, 50); // Keep last 50 songs
    });
  };

  // Update song name
  const updateSongName = (songId, newName) => {
    setSongHistory(prev => prev.map(song =>
      song.id === songId ? { ...song, name: newName } : song
    ));
  };

  // Delete song from history
  const deleteSong = (songId) => {
    if (!confirm('Remove this song from history?')) return;
    setSongHistory(prev => prev.filter(song => song.id !== songId));
  };

  // Load custom models (defaults available to everyone, personal models only when authenticated)
  useEffect(() => {
    loadCustomModels();
  }, [auth?.isAuthenticated, auth?.token]);

  const loadCustomModels = async () => {
    try {
      // Always fetch - API returns defaults for everyone, plus personal models if authenticated
      const headers = {};
      if (auth?.token) {
        headers['Authorization'] = `Bearer ${auth.token}`;
      }
      const response = await fetch(`${API_BASE_URL}/api/custom-models`, { headers });
      if (response.ok) {
        const data = await response.json();
        setCustomModels(data.models || []);
      } else {
        // If error, still try to show empty (no crash)
        setCustomModels([]);
      }
    } catch (err) {
      console.error('Failed to load custom models:', err);
      setCustomModels([]);
    }
  };

  // Poll for job status
  const jobStatus = useJobPolling(
    currentJobId,
    (data) => {
      if (data.status === 'completed') {
        setJobComplete(true);
        setOutputUrl(data.output_url);
        if (data.lyrics) setGeneratedLyrics(data.lyrics);

        // Add to song history
        addSongToHistory({
          job_id: currentJobId,
          output_url: data.output_url,
          type: activeTab,
          prompt: data.prompt || null,
        });
      } else if (data.status === 'failed') {
        setJobFailed(true);
        setError(data.error);
      }
      setLoading(false);
    },
    (lyrics) => {
      if (lyrics) setGeneratedLyrics(lyrics);
    }
  );

  const resetState = () => {
    setCurrentJobId(null);
    setJobComplete(false);
    setJobFailed(false);
    setOutputUrl(null);
    setError(null);
  };

  const handleMusicSubmit = async (data) => {
    resetState();
    setLoading(true);
    setGeneratedLyrics('');

    try {
      const response = await fetchApi('/api/generate/music', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
      });
      setCurrentJobId(response.job_id);
    } catch (err) {
      setError(err.message);
      setLoading(false);
    }
  };

  const handleCoverSubmit = async (formData) => {
    resetState();
    setLoading(true);

    try {
      const headers = {};
      if (auth?.token) {
        headers['Authorization'] = `Bearer ${auth.token}`;
      }

      const response = await fetch(`${API_BASE_URL}/api/generate/cover`, {
        method: 'POST',
        headers,
        body: formData
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'API error');
      }

      const data = await response.json();
      setCurrentJobId(data.job_id);
    } catch (err) {
      setError(err.message);
      setLoading(false);
    }
  };

  const handleCombinedSubmit = async (formData) => {
    resetState();
    setLoading(true);

    try {
      const headers = {};
      if (auth?.token) {
        headers['Authorization'] = `Bearer ${auth.token}`;
      }

      const response = await fetch(`${API_BASE_URL}/api/generate/combined`, {
        method: 'POST',
        headers,
        body: formData
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'API error');
      }

      const data = await response.json();
      setCurrentJobId(data.job_id);
    } catch (err) {
      setError(err.message);
      setLoading(false);
    }
  };

  const handleUploadModel = async (data) => {
    // Save model from URL
    setLoading(true);
    setError(null);
    setTrainingStatus(null);

    try {
      const headers = { 'Content-Type': 'application/json' };
      if (auth?.token) {
        headers['Authorization'] = `Bearer ${auth.token}`;
      }

      const response = await fetch(`${API_BASE_URL}/api/custom-models/save-url`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          model_name: data.model_name,
          model_url: data.model_url
        })
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Save failed');
      }

      await loadCustomModels();
      setError(null);
      alert('✅ Model saved successfully!');
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const [trainingStatus, setTrainingStatus] = useState(null);
  const [trainingJobId, setTrainingJobId] = useState(null);

  // Poll for training status
  useEffect(() => {
    if (!trainingJobId) return;

    const pollInterval = setInterval(async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/training-jobs/${trainingJobId}`);
        const data = await response.json();

        setTrainingStatus({
          status: data.status,
          progress: data.progress || 0,
          message: data.message || 'Training in progress...'
        });

        if (data.status === 'completed') {
          clearInterval(pollInterval);
          setLoading(false);
          setTrainingJobId(null);
          await loadCustomModels();
          alert('✅ Training completed! Your model is now available.');
        } else if (data.status === 'failed') {
          clearInterval(pollInterval);
          setLoading(false);
          setTrainingJobId(null);
          setError(data.error || 'Training failed');
        }
      } catch (error) {
        console.error('Training poll error:', error);
      }
    }, 5000);

    return () => clearInterval(pollInterval);
  }, [trainingJobId]);

  const handleTrainModel = async (formData) => {
    setLoading(true);
    setError(null);
    setTrainingStatus({ status: 'processing', progress: 0, message: 'Starting training...' });

    try {
      const headers = {};
      if (auth?.token) {
        headers['Authorization'] = `Bearer ${auth.token}`;
      }

      const response = await fetch(`${API_BASE_URL}/api/custom-models/train`, {
        method: 'POST',
        headers,
        body: formData
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Training failed to start');
      }

      const data = await response.json();
      setTrainingJobId(data.job_id);
      setTrainingStatus({ status: 'processing', progress: 5, message: 'Training job started...' });
    } catch (err) {
      setError(err.message);
      setLoading(false);
      setTrainingStatus({ status: 'failed', progress: 0, message: err.message });
    }
  };

  const handleDeleteModel = async (modelId) => {
    if (!confirm('Are you sure you want to delete this model?')) return;

    try {
      const headers = {};
      if (auth?.token) {
        headers['Authorization'] = `Bearer ${auth.token}`;
      }

      const response = await fetch(`${API_BASE_URL}/api/custom-models/${modelId}`, {
        method: 'DELETE',
        headers
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Delete failed');
      }

      await loadCustomModels();
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <div style={styles.app}>
      {/* Hamburger Menu Button (Only when authed) */}
      {auth.isAuthenticated && (
        <HamburgerButton
          onClick={() => setSidebarOpen(true)}
          songCount={songHistory.length}
        />
      )}

      {/* Song History Sidebar (Only when authed) */}
      {auth.isAuthenticated && (
        <SongHistorySidebar
          isOpen={sidebarOpen}
          onClose={() => setSidebarOpen(false)}
          songs={songHistory}
          onUpdateSong={updateSongName}
          onDeleteSong={deleteSong}
        />
      )}

      <Header />

      <main style={styles.main}>
        <TabNav />

        <div style={styles.content}>
          {error && !loading && !jobFailed && (
            <div style={styles.errorBanner}>
              ⚠️ {error}
              <button onClick={() => setError(null)} style={styles.errorClose}>×</button>
            </div>
          )}

          {activeTab === 'music' && (
            <div style={styles.tabContent}>
              <h2 style={styles.tabTitle}>Generate Original Music</h2>
              <p style={styles.tabDescription}>
                Create unique songs from text descriptions
              </p>
              <MusicGenerationForm
                onSubmit={handleMusicSubmit}
                loading={loading}
                generatedLyrics={generatedLyrics}
                onLyricsChange={setGeneratedLyrics}
              />
            </div>
          )}

          {activeTab === 'cover' && (
            <div style={styles.tabContent}>
              <h2 style={styles.tabTitle}>Create Voice Cover</h2>
              <p style={styles.tabDescription}>
                Transform songs with different voices using RVC voice conversion
              </p>
              <VoiceCoverForm
                onSubmit={handleCoverSubmit}
                loading={loading}
                customModels={customModels}
              />
            </div>
          )}

          {activeTab === 'combined' && (
            <div style={styles.tabContent}>
              <h2 style={styles.tabTitle}>Original Music + Your Voice</h2>
              <p style={styles.tabDescription}>
                Generate a unique song and have it sung in your own voice
              </p>
              <CombinedForm
                onSubmit={handleCombinedSubmit}
                loading={loading}
                customModels={customModels}
                onGoToVoiceModels={() => navigate('/voice-models')}
              />
            </div>
          )}

          {activeTab === 'train' && (
            <div style={styles.tabContent}>
              <h2 style={styles.tabTitle}>Voice Models</h2>
              <p style={styles.tabDescription}>
                Train a custom AI model from your voice recordings to create personalized covers
              </p>
              <TrainModelForm
                onTrainSubmit={handleTrainModel}
                onUploadSubmit={handleUploadModel}
                loading={loading}
                customModels={customModels}
                onDeleteModel={handleDeleteModel}
                trainingStatus={trainingStatus}
              />
            </div>
          )}

          <AnimatedProgress
            isActive={loading}
            isComplete={jobComplete}
            isFailed={jobFailed}
            outputUrl={outputUrl}
            lyrics={generatedLyrics}
            error={error}
          />
        </div>

        <GenerationHistory />
      </main>

      <footer style={styles.footer}>
        <p>AI Music Generator | ♥ Teoman Yamacı ♥ | Scate AI Case Study </p>
      </footer>

      {/* Auth Modal for /login and /signup routes */}
      <AuthModal
        isOpen={showAuthModal}
        onClose={handleAuthModalClose}
        initialMode={authMode}
      />

      {/* Redirect / to /music */}
      {location.pathname === '/' && <Navigate to="/music" replace />}
    </div>
  );
}

// Wrap with AuthProvider and BrowserRouter
export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <AppContent />
      </AuthProvider>
    </BrowserRouter>
  );
}

// ============================================================================
// Styles
// ============================================================================

const styles = {
  app: {
    minHeight: '100vh',
    background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)',
    color: '#fff',
    fontFamily: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
  },
  header: {
    padding: '30px 20px',
    borderBottom: '1px solid rgba(255,255,255,0.1)',
  },
  headerContent: {
    maxWidth: '1200px',
    margin: '0 auto',
    textAlign: 'center',
  },
  logo: {
    fontSize: '2.5rem',
    margin: 0,
    background: 'linear-gradient(90deg, #ff6b35, #f7c59f)',
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
    backgroundClip: 'text',
  },
  tagline: {
    color: '#aaa',
    marginTop: '8px',
  },
  main: {
    maxWidth: '900px',
    margin: '0 auto',
    padding: '20px',
  },
  tabNav: {
    display: 'flex',
    gap: '10px',
    marginBottom: '30px',
    flexWrap: 'wrap',
  },
  tab: {
    flex: 1,
    minWidth: '150px',
    padding: '15px 20px',
    background: 'rgba(255,255,255,0.05)',
    border: '1px solid rgba(255,255,255,0.1)',
    borderRadius: '12px',
    color: '#aaa',
    cursor: 'pointer',
    transition: 'all 0.3s ease',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: '5px',
  },
  tabActive: {
    background: 'rgba(255,107,53,0.2)',
    borderColor: '#ff6b35',
    color: '#fff',
  },
  tabLabel: {
    fontSize: '1rem',
    fontWeight: '600',
  },
  tabFlow: {
    fontSize: '0.75rem',
    opacity: 0.7,
  },
  content: {
    background: 'rgba(255,255,255,0.05)',
    borderRadius: '16px',
    padding: '30px',
    border: '1px solid rgba(255,255,255,0.1)',
  },
  tabContent: {},
  tabTitle: {
    fontSize: '1.5rem',
    marginBottom: '8px',
  },
  tabDescription: {
    color: '#aaa',
    marginBottom: '25px',
  },
  form: {
    display: 'flex',
    flexDirection: 'column',
    gap: '20px',
  },
  formGroup: {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
  },
  formRow: {
    display: 'flex',
    gap: '20px',
    flexWrap: 'wrap',
  },
  label: {
    fontSize: '0.9rem',
    color: '#ccc',
  },
  textarea: {
    padding: '12px',
    background: 'rgba(255,255,255,0.05)',
    border: '1px solid rgba(255,255,255,0.2)',
    borderRadius: '8px',
    color: '#fff',
    fontSize: '1rem',
    resize: 'vertical',
    fontFamily: 'inherit',
  },
  select: {
    padding: '12px',
    background: 'rgba(255,255,255,0.05)',
    border: '1px solid rgba(255,255,255,0.2)',
    borderRadius: '8px',
    color: '#fff',
    fontSize: '1rem',
  },
  input: {
    padding: '12px',
    background: 'rgba(255,255,255,0.05)',
    border: '1px solid rgba(255,255,255,0.2)',
    borderRadius: '8px',
    color: '#fff',
    fontSize: '1rem',
  },
  radioGroup: {
    display: 'flex',
    gap: '20px',
    flexWrap: 'wrap',
  },
  radioLabel: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    color: '#ccc',
    cursor: 'pointer',
    padding: '10px 15px',
    background: 'rgba(255,255,255,0.03)',
    borderRadius: '8px',
    border: '1px solid rgba(255,255,255,0.1)',
    transition: 'all 0.2s',
  },
  radio: {
    accentColor: '#ff6b35',
  },
  helpText: {
    fontSize: '0.8rem',
    color: '#888',
    marginTop: '4px',
  },
  infoBox: {
    padding: '15px',
    background: 'rgba(96, 165, 250, 0.1)',
    border: '1px solid rgba(96, 165, 250, 0.3)',
    borderRadius: '8px',
    marginBottom: '10px',
  },
  slider: {
    width: '100%',
    accentColor: '#ff6b35',
  },
  checkbox: {
    accentColor: '#ff6b35',
  },
  checkboxLabel: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    color: '#ccc',
    cursor: 'pointer',
  },
  fileInput: {
    padding: '12px',
    background: 'rgba(255,255,255,0.05)',
    border: '1px solid rgba(255,255,255,0.2)',
    borderRadius: '8px',
    color: '#fff',
  },
  input: {
    width: '100%',
    padding: '12px 16px',
    background: 'rgba(255,255,255,0.05)',
    border: '1px solid rgba(255,255,255,0.2)',
    borderRadius: '8px',
    color: '#fff',
    fontSize: '1rem',
    boxSizing: 'border-box',
  },
  fileName: {
    fontSize: '0.85rem',
    color: '#4ade80',
    margin: 0,
  },
  submitButton: {
    padding: '15px 30px',
    background: 'linear-gradient(90deg, #ff6b35, #ff8c42)',
    border: 'none',
    borderRadius: '8px',
    color: '#fff',
    fontSize: '1.1rem',
    fontWeight: '600',
    cursor: 'pointer',
    transition: 'transform 0.2s, opacity 0.2s',
    marginTop: '10px',
  },
  promptSelector: {
    marginBottom: '10px',
  },
  promptGrid: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: '8px',
    marginTop: '8px',
  },
  promptButton: {
    padding: '8px 12px',
    background: 'rgba(255,107,53,0.1)',
    border: '1px solid rgba(255,107,53,0.3)',
    borderRadius: '6px',
    color: '#ff8c42',
    fontSize: '0.85rem',
    cursor: 'pointer',
    transition: 'all 0.2s',
  },
  progressContainer: {
    marginTop: '30px',
    padding: '20px',
    background: 'rgba(255,255,255,0.03)',
    borderRadius: '12px',
    border: '1px solid rgba(255,255,255,0.1)',
  },
  progressHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '15px',
  },
  progressTitle: {
    fontSize: '1.1rem',
    fontWeight: '600',
  },
  progressStatus: {
    fontSize: '1.1rem',
    fontWeight: '600',
  },
  progressBar: {
    height: '8px',
    background: 'rgba(255,255,255,0.1)',
    borderRadius: '4px',
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    borderRadius: '4px',
    transition: 'width 0.5s ease',
  },
  resultContainer: {
    marginTop: '20px',
    display: 'flex',
    flexDirection: 'column',
    gap: '15px',
  },
  audioPlayer: {
    width: '100%',
    borderRadius: '8px',
  },
  lyricsDisplay: {
    padding: '15px',
    background: 'rgba(255,255,255,0.03)',
    borderRadius: '8px',
    border: '1px solid rgba(255,255,255,0.1)',
  },
  lyricsText: {
    margin: 0,
    whiteSpace: 'pre-wrap',
    fontFamily: 'inherit',
    fontSize: '0.9rem',
    color: '#ccc',
    lineHeight: 1.6,
  },
  downloadButton: {
    display: 'inline-block',
    padding: '12px 24px',
    background: '#4ade80',
    borderRadius: '8px',
    color: '#000',
    textDecoration: 'none',
    fontWeight: '600',
    textAlign: 'center',
  },
  errorText: {
    color: '#f87171',
    marginTop: '10px',
  },
  errorBanner: {
    padding: '15px',
    background: 'rgba(248,113,113,0.1)',
    border: '1px solid #f87171',
    borderRadius: '8px',
    color: '#f87171',
    marginBottom: '20px',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  errorClose: {
    background: 'none',
    border: 'none',
    color: '#f87171',
    fontSize: '1.5rem',
    cursor: 'pointer',
  },
  historyContainer: {
    marginTop: '30px',
    padding: '20px',
    background: 'rgba(255,255,255,0.03)',
    borderRadius: '12px',
    border: '1px solid rgba(255,255,255,0.1)',
  },
  historyTitle: {
    fontSize: '1.1rem',
    marginBottom: '15px',
    color: '#fff',
  },
  jobsList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '10px',
  },
  jobItem: {
    padding: '12px',
    background: 'rgba(255,255,255,0.02)',
    borderRadius: '8px',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    flexWrap: 'wrap',
    gap: '10px',
  },
  jobInfo: {
    display: 'flex',
    gap: '10px',
  },
  jobFlow: {
    padding: '4px 8px',
    background: 'rgba(255,107,53,0.2)',
    borderRadius: '4px',
    fontSize: '0.8rem',
  },
  jobStatus: {
    padding: '4px 8px',
    background: 'rgba(255,255,255,0.1)',
    borderRadius: '4px',
    fontSize: '0.8rem',
  },
  miniPlayer: {
    height: '32px',
    maxWidth: '200px',
  },
  footer: {
    textAlign: 'center',
    padding: '30px',
    color: '#666',
    borderTop: '1px solid rgba(255,255,255,0.1)',
  },
};
