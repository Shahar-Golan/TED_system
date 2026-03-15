import { useState, useEffect } from 'react';
import { chatAPI } from '../services/api';
import './SpeakerProfile.css';

function SpeakerProfile({ onBack }) {
  const [speakers, setSpeakers] = useState([]);
  const [selectedProfile, setSelectedProfile] = useState(null);
  const [loading, setLoading] = useState(true);
  const [profileLoading, setProfileLoading] = useState(false);
  const [activeSection, setActiveSection] = useState('bio');

  useEffect(() => {
    loadSpeakers();
  }, []);

  const loadSpeakers = async () => {
    try {
      const data = await chatAPI.getSpeakers();
      setSpeakers(data);
    } catch (err) {
      console.error('Failed to load speakers:', err);
    } finally {
      setLoading(false);
    }
  };

  const selectSpeaker = async (speakerId) => {
    setProfileLoading(true);
    setActiveSection('bio');
    try {
      const profile = await chatAPI.getSpeakerProfile(speakerId);
      setSelectedProfile(profile);
    } catch (err) {
      console.error('Failed to load profile:', err);
    } finally {
      setProfileLoading(false);
    }
  };

  const sections = [
    { id: 'bio', label: 'Biography' },
    { id: 'topics', label: 'Notable Topics' },
    { id: 'timeline', label: 'Timeline' },
    { id: 'controversies', label: 'Controversies' },
    { id: 'relationships', label: 'Relationships' },
    { id: 'perception', label: 'Public Perception' },
    { id: 'media', label: 'Media Profile' },
    { id: 'data', label: 'Dataset Insights' },
    { id: 'news', label: 'Recent News' },
  ];

  const renderBio = (profile) => {
    const bio = profile.bio || {};
    return (
      <div className="sp-section">
        <div className="sp-bio-grid">
          {[
            ['Full Name', bio.full_name],
            ['Born', bio.born],
            ['Party', bio.party],
            ['Current Role', bio.current_role],
            ['Net Worth', bio.net_worth_estimate],
          ].map(([label, value]) => (
            <div key={label} className="sp-bio-item">
              <span className="sp-bio-label">{label}</span>
              <span className="sp-bio-value">{value || 'N/A'}</span>
            </div>
          ))}
        </div>
        {bio.previous_roles && (
          <div className="sp-subsection">
            <h4>Previous Roles</h4>
            <ul className="sp-list">
              {bio.previous_roles.map((role, i) => <li key={i}>{role}</li>)}
            </ul>
          </div>
        )}
        {bio.education && (
          <div className="sp-subsection">
            <h4>Education</h4>
            <ul className="sp-list">
              {bio.education.map((edu, i) => <li key={i}>{edu}</li>)}
            </ul>
          </div>
        )}
      </div>
    );
  };

  const renderTopics = (profile) => {
    const topics = profile.notable_topics || [];
    return (
      <div className="sp-section">
        {topics.map((topic, i) => (
          <div key={i} className="sp-topic-card">
            <div className="sp-topic-header">
              <h4>{topic.topic}</h4>
              <span className="sp-category-badge">{topic.category}</span>
            </div>
            <p className="sp-detail"><strong>Stance:</strong> {topic.stance}</p>
            {topic.key_statements && (
              <div className="sp-quotes">
                {topic.key_statements.map((stmt, j) => (
                  <blockquote key={j}>{stmt}</blockquote>
                ))}
              </div>
            )}
            {topic.evolution && <p className="sp-detail"><strong>Evolution:</strong> {topic.evolution}</p>}
            {topic.controversies && <p className="sp-detail"><strong>Controversies:</strong> {topic.controversies}</p>}
          </div>
        ))}
      </div>
    );
  };

  const renderTimeline = (profile) => {
    const events = profile.timeline_highlights || [];
    return (
      <div className="sp-section">
        <div className="sp-timeline">
          {events.map((event, i) => (
            <div key={i} className="sp-timeline-item">
              <div className="sp-timeline-dot" />
              <div className="sp-timeline-year">{event.year}</div>
              <div className="sp-timeline-content">
                <h4>{event.event}</h4>
                <p>{event.significance}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  const renderControversies = (profile) => {
    const items = profile.controversies || [];
    return (
      <div className="sp-section">
        {items.map((item, i) => (
          <div key={i} className="sp-controversy-card">
            <div className="sp-controversy-header">
              <h4>{item.title}</h4>
              <span className="sp-year-badge">{item.year}</span>
            </div>
            <p>{item.description}</p>
            {item.outcome && <p className="sp-detail"><strong>Outcome:</strong> {item.outcome}</p>}
            {item.impact && <p className="sp-detail"><strong>Impact:</strong> {item.impact}</p>}
          </div>
        ))}
      </div>
    );
  };

  const renderRelationships = (profile) => {
    const rel = profile.relationships || {};
    return (
      <div className="sp-section">
        <div className="sp-rel-grid">
          {rel.allies && (
            <div className="sp-rel-group">
              <h4>Allies</h4>
              <ul className="sp-list">{rel.allies.map((a, i) => <li key={i}>{a}</li>)}</ul>
            </div>
          )}
          {rel.opponents && (
            <div className="sp-rel-group">
              <h4>Opponents</h4>
              <ul className="sp-list">{rel.opponents.map((o, i) => <li key={i}>{o}</li>)}</ul>
            </div>
          )}
        </div>
        {rel.co_mentioned_figures && (
          <div className="sp-subsection">
            <h4>Co-mentioned Figures (from dataset)</h4>
            <div className="sp-co-mentions">
              {Object.entries(rel.co_mentioned_figures).map(([name, count]) => (
                <div key={name} className="sp-co-mention-item">
                  <span className="sp-co-name">{name}</span>
                  <span className="sp-co-count">{count.toLocaleString()} articles</span>
                </div>
              ))}
            </div>
          </div>
        )}
        {rel.relationship_context && (
          <div className="sp-subsection">
            <h4>Context</h4>
            <p>{rel.relationship_context}</p>
          </div>
        )}
      </div>
    );
  };

  const renderPerception = (profile) => {
    const pp = profile.public_perception || {};
    return (
      <div className="sp-section">
        {[
          ['Approval Trend', pp.approval_trend],
          ['Base Support', pp.base_support],
          ['Opposition', pp.opposition],
        ].map(([title, text]) => text && (
          <div key={title} className="sp-subsection">
            <h4>{title}</h4>
            <p>{text}</p>
          </div>
        ))}
        {pp.key_narratives && (
          <div className="sp-subsection">
            <h4>Key Narratives</h4>
            <ul className="sp-list">{pp.key_narratives.map((n, i) => <li key={i}>{n}</li>)}</ul>
          </div>
        )}
      </div>
    );
  };

  const renderMedia = (profile) => {
    const mp = profile.media_profile || {};
    return (
      <div className="sp-section">
        {mp.coverage_volume && (
          <div className="sp-subsection">
            <h4>Coverage Volume</h4>
            <p>{mp.coverage_volume}</p>
          </div>
        )}
        {mp.top_covering_states && (
          <div className="sp-subsection">
            <h4>Top Covering States</h4>
            <div className="sp-state-bars">
              {Object.entries(mp.top_covering_states).map(([state, count]) => {
                const max = Math.max(...Object.values(mp.top_covering_states));
                return (
                  <div key={state} className="sp-state-bar">
                    <span className="sp-state-name">{state}</span>
                    <div className="sp-bar-track">
                      <div className="sp-bar-fill" style={{ width: `${(count / max) * 100}%` }} />
                    </div>
                    <span className="sp-state-count">{count.toLocaleString()}</span>
                  </div>
                );
              })}
            </div>
          </div>
        )}
        {mp.media_narrative && (
          <div className="sp-subsection"><h4>Media Narrative</h4><p>{mp.media_narrative}</p></div>
        )}
        {mp.sentiment_trend && (
          <div className="sp-subsection"><h4>Sentiment Trend</h4><p>{mp.sentiment_trend}</p></div>
        )}
      </div>
    );
  };

  const renderRecentNews = (profile) => {
    const rn = profile.recent_news;
    if (!rn || !rn.items || rn.items.length === 0) {
      return (
        <div className="sp-section">
          <p className="sp-empty-section">No recent news available for this speaker.</p>
        </div>
      );
    }
    return (
      <div className="sp-section">
        <div className="sp-news-meta">
          {rn.summary && <p className="sp-news-summary">{rn.summary}</p>}
          <span className="sp-news-updated">
            Updated: {rn.last_updated ? new Date(rn.last_updated).toLocaleDateString() : 'unknown'}
            {rn.date_range ? ` · ${rn.date_range}` : ''}
          </span>
        </div>
        {rn.items.map((item, i) => (
          <div key={i} className="sp-news-card">
            <div className="sp-news-header">
              <h4>{item.headline}</h4>
              <div className="sp-news-badges">
                <span className="sp-news-date-badge">{item.date}</span>
                {item.significance && (
                  <span className="sp-news-sig-badge">{item.significance}</span>
                )}
              </div>
            </div>
            <p>{item.summary}</p>
          </div>
        ))}
      </div>
    );
  };

  const renderDataInsights = (profile) => {
    const di = profile.dataset_insights || {};
    return (
      <div className="sp-section">
        <div className="sp-data-stats">
          <div className="sp-stat-card">
            <div className="sp-stat-number">{(di.total_articles || 0).toLocaleString()}</div>
            <div className="sp-stat-label">Articles</div>
          </div>
          <div className="sp-stat-card">
            <div className="sp-stat-number">{di.date_range || 'N/A'}</div>
            <div className="sp-stat-label">Date Range</div>
          </div>
        </div>
        {di.top_title_keywords && (
          <div className="sp-subsection">
            <h4>Top Title Keywords</h4>
            <div className="sp-keywords">
              {Object.entries(di.top_title_keywords).map(([word, count]) => (
                <span key={word} className="sp-keyword">{word} <small>({count})</small></span>
              ))}
            </div>
          </div>
        )}
        {di.geographic_focus && (
          <div className="sp-subsection"><h4>Geographic Focus</h4><p>{di.geographic_focus}</p></div>
        )}
      </div>
    );
  };

  const renderSection = () => {
    if (!selectedProfile) return null;
    const renderers = { bio: renderBio, topics: renderTopics, timeline: renderTimeline, controversies: renderControversies, relationships: renderRelationships, perception: renderPerception, media: renderMedia, data: renderDataInsights, news: renderRecentNews };
    return renderers[activeSection]?.(selectedProfile) || null;
  };

  if (loading) return <div className="sp-loading">Loading speakers...</div>;

  return (
    <div className="sp-container">
      <div className="sp-header">
        <button className="sp-back-btn" onClick={onBack}>Back</button>
        <h2>Speaker Profiles</h2>
      </div>

      <div className="sp-layout">
        <div className="sp-sidebar">
          {speakers.map((s) => (
            <div
              key={s.speaker_id}
              className={`sp-speaker-card ${selectedProfile?.name === s.name ? 'active' : ''}`}
              onClick={() => selectSpeaker(s.speaker_id)}
            >
              <div className="sp-speaker-name">{s.name}</div>
              <div className="sp-speaker-meta">{s.party}</div>
              <div className="sp-speaker-articles">{s.total_articles.toLocaleString()} articles</div>
            </div>
          ))}
        </div>

        <div className="sp-content">
          {profileLoading ? (
            <div className="sp-loading">Loading profile...</div>
          ) : selectedProfile ? (
            <>
              <div className="sp-profile-header">
                <h3>{selectedProfile.name}</h3>
                <p className="sp-role">{selectedProfile.bio?.current_role}</p>
              </div>
              <div className="sp-section-tabs">
                {sections.map((s) => (
                  <button
                    key={s.id}
                    className={`sp-tab ${activeSection === s.id ? 'active' : ''}`}
                    onClick={() => setActiveSection(s.id)}
                  >{s.label}</button>
                ))}
              </div>
              <div className="sp-section-content">{renderSection()}</div>
            </>
          ) : (
            <div className="sp-empty">Select a speaker to view their profile</div>
          )}
        </div>
      </div>
    </div>
  );
}

export default SpeakerProfile;
