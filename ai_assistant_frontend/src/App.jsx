import { useState } from 'react'
import './App.css'

const API_URL = 'http://localhost:8000'

function App() {
  const [query, setQuery] = useState('')
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

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
        <h1>AI Assistant</h1>
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
                    <div key={index} className="video-card">
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
                          {video.link ? (
                            <a 
                              href={video.link} 
                              target="_blank" 
                              rel="noopener noreferrer"
                              className="video-link"
                            >
                              {video.title || 'Untitled'}
                            </a>
                          ) : (
                            video.title || 'Untitled'
                          )}
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
    </div>
  )
}

export default App

