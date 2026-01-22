import { useState, useEffect, useRef } from 'react'
import './App.css'

const API_URL = 'http://localhost:8000'

// Utility function to extract YouTube video ID from URL
const extractYouTubeVideoId = (url) => {
  if (!url) return null
  
  const patterns = [
    /(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)/,
    /^([a-zA-Z0-9_-]{11})$/
  ]
  
  for (const pattern of patterns) {
    const match = url.match(pattern)
    if (match) return match[1]
  }
  
  return null
}

// VideoModal Component
const VideoModal = ({ video, isOpen, onClose }) => {
  const modalRef = useRef(null)
  const closeButtonRef = useRef(null)

  useEffect(() => {
    if (isOpen) {
      // Prevent body scroll when modal is open
      document.body.style.overflow = 'hidden'
      // Focus close button for accessibility
      closeButtonRef.current?.focus()
    } else {
      document.body.style.overflow = ''
    }

    // Cleanup: restore scroll when component unmounts
    return () => {
      document.body.style.overflow = ''
    }
  }, [isOpen])

  useEffect(() => {
    const handleEscape = (e) => {
      if (e.key === 'Escape' && isOpen) {
        onClose()
      }
    }

    if (isOpen) {
      document.addEventListener('keydown', handleEscape)
      return () => document.removeEventListener('keydown', handleEscape)
    }
  }, [isOpen, onClose])

  const handleBackdropClick = (e) => {
    if (e.target === modalRef.current) {
      onClose()
    }
  }

  if (!isOpen || !video) return null

  const videoId = extractYouTubeVideoId(video.link)
  const embedUrl = videoId 
    ? `https://www.youtube.com/embed/${videoId}?autoplay=1`
    : null

  return (
    <div 
      className="modal-overlay" 
      ref={modalRef}
      onClick={handleBackdropClick}
      role="dialog"
      aria-modal="true"
      aria-labelledby="modal-title"
    >
      <div className="modal-content">
        <button
          ref={closeButtonRef}
          className="modal-close"
          onClick={onClose}
          aria-label="Close modal"
        >
          ×
        </button>
        
        {embedUrl ? (
          <div className="modal-video-wrapper">
            <iframe
              src={embedUrl}
              title={video.title || 'Video player'}
              className="modal-video-iframe"
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
              allowFullScreen
            />
          </div>
        ) : (
          <div className="modal-video-error">
            <p>Unable to load video player</p>
          </div>
        )}

        <div className="modal-footer">
          <a
            href={video.link}
            target="_blank"
            rel="noopener noreferrer"
            className="modal-youtube-link"
            aria-label="Watch on YouTube"
          >
            <svg 
              className="youtube-icon" 
              viewBox="0 0 24 24" 
              fill="currentColor"
              aria-hidden="true"
            >
              <path d="M23.498 6.186a3.016 3.016 0 0 0-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.377.505A3.017 3.017 0 0 0 .502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 0 0 2.122 2.136c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 0 0 2.122-2.136C24 15.93 24 12 24 12s0-3.93-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z"/>
            </svg>
            Watch on YouTube
          </a>
        </div>
      </div>
    </div>
  )
}

function App() {
  const [query, setQuery] = useState('')
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [selectedVideo, setSelectedVideo] = useState(null)

  const handleSubmit = async (e) => {
    e.preventDefault()
    
    if (!query.trim()) {
      return
    }

    setLoading(true)
    setError(null)
    setData(null)

    try {
      const response = await fetch(`${API_URL}/?query=${encodeURIComponent(query)}`)
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const result = await response.json()
      setData(result)
    } catch (err) {
      setError(err.message || 'Failed to fetch data. Please make sure the backend is running.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <header className="header">
        <h1>YopeAI Assistant</h1>
      </header>

      <main className="main">
        <form onSubmit={handleSubmit} className="search-form">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Enter your query..."
            className="search-input"
            disabled={loading}
          />
          <button 
            type="submit" 
            className="search-button"
            disabled={loading}
          >
            {loading ? 'Searching...' : 'Search'}
          </button>
        </form>

        {loading && (
          <div className="loading">
            <div className="spinner"></div>
            <p>Loading...</p>
          </div>
        )}

        {error && (
          <div className="error">
            <p>Error: {error}</p>
          </div>
        )}

        {data && !loading && (
          <div className="results">
            {data.explain && (
              <section className="explain-section">
                <h2>Explanation</h2>
                <p className="explain-text">{data.explain}</p>
              </section>
            )}

            {data.videos && data.videos.length > 0 && (
              <section className="videos-section">
                <h2>Videos</h2>
                <div className="videos-grid">
                  {data.videos.map((video, index) => (
                    <div 
                      key={index} 
                      className="video-card"
                      onClick={() => setSelectedVideo(video)}
                      role="button"
                      tabIndex={0}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' || e.key === ' ') {
                          e.preventDefault()
                          setSelectedVideo(video)
                        }
                      }}
                      aria-label={`Play video: ${video.title || 'Untitled'}`}
                    >
                      {video.thumbnails && video.thumbnails.length > 0 && (
                        <img 
                          src={video.thumbnails[0]} 
                          alt={video.title || 'Video thumbnail'}
                          className="video-thumbnail"
                          onError={(e) => {
                            e.target.style.display = 'none'
                          }}
                        />
                      )}
                      <div className="video-content">
                        <h3 className="video-title">
                          {video.title || 'Untitled'}
                        </h3>
                        {video.description && (
                          <p className="video-description">{video.description}</p>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </section>
            )}

            {data.references && data.references.length > 0 && (
              <section className="references-section">
                <h2>References</h2>
                <ul className="references-list">
                  {data.references.map((ref, index) => (
                    <li key={index} className="reference-item">
                      {ref.link ? (
                        <a 
                          href={ref.link} 
                          target="_blank" 
                          rel="noopener noreferrer"
                          className="reference-link"
                        >
                          <strong>{ref.title || 'Untitled'}</strong>
                        </a>
                      ) : (
                        <strong>{ref.title || 'Untitled'}</strong>
                      )}
                      {ref.description && (
                        <p className="reference-description">{ref.description}</p>
                      )}
                    </li>
                  ))}
                </ul>
              </section>
            )}
          </div>
        )}
      </main>

      <VideoModal
        video={selectedVideo}
        isOpen={!!selectedVideo}
        onClose={() => setSelectedVideo(null)}
      />
    </div>
  )
}

export default App

