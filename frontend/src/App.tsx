import React, { useState, useEffect, useRef } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { 
  MessageSquare, 
  Send, 
  Database, 
  Activity, 
  GitFork, 
  AlertCircle, 
  Trash2, 
  Compass,
  FileText,
  Network,
  Settings
} from 'lucide-react';
import './App.css';
import Pipelines from './Pipelines';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  query_type?: string;
  sources?: any[];
  graph_paths?: string[];
  graph_results?: any;
  vector_results?: any[];
}

function App() {
  const [currentView, setCurrentView] = useState<'chat' | 'pipelines'>('chat');
  const [question, setQuestion] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  
  // Tabs: 'answer' | 'vector' | 'graph'
  const [activeTab, setActiveTab] = useState<'answer' | 'vector' | 'graph'>('answer');
  const [selectedResponse, setSelectedResponse] = useState<Message | null>(null);
  
  // Health State
  const [health, setHealth] = useState({ status: 'unknown', qdrant: 'unknown', neo4j: 'unknown' });
  
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Refs & states for handling force graph centering and resizing
  const fgRef = useRef<any>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 400, height: 400 });

  // Fetch health check on mount
  useEffect(() => {
    fetchHealth();
  }, []);

  const fetchHealth = async () => {
    try {
      const res = await fetch('http://127.0.0.1:8000/health');
      const data = await res.json();
      setHealth(data);
    } catch (e) {
      console.error('Failed to fetch health check', e);
      setHealth({ status: 'unhealthy', qdrant: 'offline', neo4j: 'offline' });
    }
  };

  // Scroll to bottom of chat list
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim() || isLoading) return;

    const userQuestion = question.trim();
    setQuestion('');
    
    // Add User Message
    const userMsg: Message = { role: 'user', content: userQuestion };
    const updatedMessages = [...messages, userMsg];
    setMessages(updatedMessages);
    setIsLoading(true);

    // Format chat history
    const history = messages.map(msg => ({
      role: msg.role,
      content: msg.content
    }));

    try {
      const response = await fetch('http://127.0.0.1:8000/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: userQuestion, history })
      });

      if (!response.ok) throw new Error('API server returned error');
      
      const data = await response.json();
      
      // Add Assistant Message
      const assistantMsg: Message = {
        role: 'assistant',
        content: data.answer,
        query_type: data.query_type,
        sources: data.sources,
        graph_paths: data.graph_paths,
        graph_results: data.graph_results,
        vector_results: data.vector_results || []
      };

      setMessages(prev => [...prev, assistantMsg]);
      setSelectedResponse(assistantMsg);
      
      // Auto switch tabs based on retrieval mode
      if (data.query_type === 'VECTOR_ONLY') {
        setActiveTab('vector');
      } else if (data.query_type === 'GRAPH_ONLY') {
        setActiveTab('graph');
      } else {
        setActiveTab('answer');
      }

    } catch (error) {
      console.error(error);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Error: Failed to process query. Please check your backend connection.'
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleClearChat = () => {
    setMessages([]);
    setSelectedResponse(null);
  };

  // Helper to color and label graph nodes based on naming rules
  const getNodeMetadata = (name: string) => {
    if (name.startsWith('INC-')) return { label: 'Incident', color: '#ef4444' };
    if (name.startsWith('CV-')) return { label: 'Component', color: '#f97316' };
    if (name.includes('Plant') || name.includes('Zone')) return { label: 'Location', color: '#3b82f6' };
    if (name.includes('Ltd') || name.includes('Inc') || name.includes('CoolTech')) return { label: 'Vendor', color: '#8b5cf6' };
    if (name.includes('Team')) return { label: 'Team', color: '#06b6d4' };
    if (name === 'John Patel' || name === 'Sarah Shah') return { label: 'Employee', color: '#14b8a6' };
    return { label: 'Asset', color: '#ec4899' };
  };

  // Format Graph results for react-force-graph-2d
  const getGraphData = () => {
    if (!selectedResponse || !selectedResponse.graph_results) return { nodes: [], links: [] };
    const { nodes, edges } = selectedResponse.graph_results;
    
    const formattedNodes = nodes.map((nodeName: string) => {
      const meta = getNodeMetadata(nodeName);
      return {
        id: nodeName,
        name: nodeName,
        label: meta.label,
        color: meta.color,
        val: 12
      };
    });

    const formattedLinks = edges.map((edge: any) => ({
      source: edge.source,
      target: edge.target,
      label: edge.type
    }));

    return { nodes: formattedNodes, links: formattedLinks };
  };

  const graphData = getGraphData();

  // Resize observer to track dimensions of graph viewer container
  useEffect(() => {
    if (!containerRef.current) return;
    const resizeObserver = new ResizeObserver((entries) => {
      for (let entry of entries) {
        const { width, height } = entry.contentRect;
        if (width > 0 && height > 0) {
          setDimensions({ width, height });
        }
      }
    });
    resizeObserver.observe(containerRef.current);
    return () => resizeObserver.disconnect();
  }, [activeTab, selectedResponse]);

  // Center and fit the graph to the canvas
  useEffect(() => {
    if (activeTab === 'graph' && graphData.nodes.length > 0) {
      const timer = setTimeout(() => {
        if (fgRef.current) {
          fgRef.current.zoomToFit(400, 30);
          fgRef.current.d3ReheatSimulation();
        }
      }, 100);
      return () => clearTimeout(timer);
    }
  }, [activeTab, selectedResponse, graphData]);

  return (
    <div className="main-layout">
      {/* SIDEBAR NAVIGATION */}
      <nav className="sidebar">
        <div className="sidebar-brand">
          <Activity size={24} className="brand-icon" />
          <span>OpsRAG</span>
        </div>
        <ul className="sidebar-nav">
          <li>
            <button 
              className={`nav-btn ${currentView === 'chat' ? 'active' : ''}`}
              onClick={() => setCurrentView('chat')}
            >
              <MessageSquare size={20} />
              <span>Operator Chat</span>
            </button>
          </li>
          <li>
            <button 
              className={`nav-btn ${currentView === 'pipelines' ? 'active' : ''}`}
              onClick={() => setCurrentView('pipelines')}
            >
              <Settings size={20} />
              <span>Pipelines</span>
            </button>
          </li>
        </ul>
      </nav>

      {/* MAIN CONTENT AREA */}
      <div className="main-content">
        {currentView === 'pipelines' ? (
          <Pipelines />
        ) : (
          <div className="app-container">
            {/* LEFT: Chat Area */}
            <div className="chat-panel">
        <div className="chat-header">
          <div className={`status-indicator ${health.status === 'healthy' ? '' : 'unhealthy'}`} />
          <div style={{ flex: 1 }}>
            <h1>Smart Operations RAG</h1>
            <p>{health.status === 'healthy' ? 'Platform Online' : 'Platform Offline'}</p>
          </div>
          {messages.length > 0 && (
            <button className="send-button" style={{ background: '#ef4444' }} onClick={handleClearChat} title="Clear Conversation">
              <Trash2 size={16} />
            </button>
          )}
        </div>

        {/* Messages List */}
        <div className="messages-list">
          {messages.length === 0 && (
            <div className="empty-state">
              <MessageSquare size={48} />
              <h2>Operator Chatbot</h2>
              <p>Ask questions about operations, incident logs, asset properties, and standard procedures (SOPs).</p>
              <div className="example-questions-grid">
                <button className="example-question-btn" onClick={() => setQuestion('How do I restart MX-200?')}>How do I restart MX-200?</button>
                <button className="example-question-btn" onClick={() => setQuestion('Who owns MX-200?')}>Who owns MX-200?</button>
                <button className="example-question-btn" onClick={() => setQuestion('Why has Plant A downtime increased?')}>Why has Plant A downtime increased?</button>
                <button className="example-question-btn" onClick={() => setQuestion('What happens if CoolTech stops supplying parts?')}>What happens if CoolTech stops supplying parts?</button>
                <button className="example-question-btn" onClick={() => setQuestion('Which supplier causes most downtime?')}>Which supplier causes most downtime?</button>
                <button className="example-question-btn" onClick={() => setQuestion('What is the SOP for Quality Checks?')}>What is the SOP for Quality Checks?</button>
              </div>
            </div>
          )}

          {messages.map((msg, index) => (
            <div key={index} className={`message-bubble ${msg.role}`}>
              {msg.content}
              {msg.role === 'assistant' && msg.query_type && (
                <div className="message-metadata" onClick={() => setSelectedResponse(msg)}>
                  <span>Retrieval:</span>
                  <span className="mode-badge">{msg.query_type}</span>
                  <span style={{ cursor: 'pointer', textDecoration: 'underline' }}>Inspect Sources</span>
                </div>
              )}
            </div>
          ))}
          {isLoading && (
            <div className="typing-indicator">
              <div className="typing-dot" />
              <div className="typing-dot" />
              <div className="typing-dot" />
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input area */}
        <div className="chat-input-area">
          <form onSubmit={handleSubmit} className="chat-input-form">
            <input
              type="text"
              className="chat-input"
              placeholder="Query the manufacturing operations..."
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              disabled={isLoading}
            />
            <button type="submit" className="send-button" disabled={!question.trim() || isLoading}>
              <Send size={16} />
            </button>
          </form>
        </div>
      </div>

      {/* RIGHT: Inspector Panel */}
      <div className="inspector-panel">
        <div className="tabs-header">
          <button 
            className={`tab-button ${activeTab === 'answer' ? 'active' : ''}`} 
            onClick={() => setActiveTab('answer')}
          >
            <Compass size={16} /> Answer & Sources
          </button>
          <button 
            className={`tab-button ${activeTab === 'vector' ? 'active' : ''}`} 
            onClick={() => setActiveTab('vector')}
          >
            <FileText size={16} /> Vector Results
          </button>
          <button 
            className={`tab-button ${activeTab === 'graph' ? 'active' : ''}`} 
            onClick={() => setActiveTab('graph')}
          >
            <Network size={16} /> Graph Viewer
          </button>
        </div>

        <div className="tab-content">
          {!selectedResponse ? (
            <div className="empty-state">
              <Activity size={36} />
              <h2>No Inspection Data</h2>
              <p>Submit a question to see the retrieval steps, vector score rankings, and graph neighborhood traversals.</p>
            </div>
          ) : (
            <>
              {/* Tab 1: Answer & Sources */}
              {activeTab === 'answer' && (
                <div className="answer-view">
                  <div className="answer-header">
                    <span>Retrieval Mode:</span>
                    <span className="mode-badge">{selectedResponse.query_type}</span>
                  </div>
                  <div className="answer-text">
                    {selectedResponse.content}
                  </div>
                  {selectedResponse.sources && selectedResponse.sources.length > 0 && (
                    <div className="sources-section">
                      <h3>Cited Context Sources:</h3>
                      <div className="source-badges">
                        {selectedResponse.sources.map((src, i) => (
                          <div key={i} className={`source-badge ${src.type}`}>
                            {src.type === 'document' ? (
                              <>
                                <FileText size={12} color="#3b82f6" /> {src.name}
                              </>
                            ) : (
                              <>
                                <Network size={12} color="#10b981" /> {src.nodes?.join(', ')}
                              </>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Tab 2: Vector Chunks */}
              {activeTab === 'vector' && (
                <div>
                  {(!selectedResponse.vector_results || selectedResponse.vector_results.length === 0) ? (
                    <div className="empty-state" style={{ height: '300px' }}>
                      <AlertCircle size={28} />
                      <p>Vector Retrieval was not executed for this query type (GRAPH_ONLY).</p>
                    </div>
                  ) : (
                    <div className="chunks-list">
                      <div className="answer-header" style={{ marginBottom: '8px' }}>
                        <Database size={12} /> Retrieved Document Chunks (top-10)
                      </div>
                      {selectedResponse.vector_results.map((chunk, i) => (
                        <div key={i} className="chunk-card">
                          <div className="chunk-header">
                            <span className="chunk-doc">
                              <FileText size={12} /> {chunk.document}
                            </span>
                            <span className="chunk-score">{(chunk.score * 100).toFixed(1)}% Match</span>
                          </div>
                          <div className="chunk-text">{chunk.text}</div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* Tab 3: Interactive Subgraph force layout */}
              {activeTab === 'graph' && (
                <div style={{ height: '100%', display: 'flex', flexDirection: 'column', gap: '16px' }}>
                  {(!selectedResponse.graph_results || selectedResponse.graph_results.nodes?.length === 0) ? (
                    <div className="empty-state" style={{ height: '300px' }}>
                      <AlertCircle size={28} />
                      <p>Graph Retrieval was not executed for this query type (VECTOR_ONLY).</p>
                    </div>
                  ) : (
                    <>
                      <div className="answer-header">
                        <GitFork size={12} /> Explored Graph Paths
                      </div>
                      {selectedResponse.graph_paths && selectedResponse.graph_paths.length > 0 && (
                        <div className="paths-list" style={{ maxHeight: '150px', overflowY: 'auto' }}>
                          {selectedResponse.graph_paths.map((path, i) => (
                            <div key={i} className="path-item">{path}</div>
                          ))}
                        </div>
                      )}

                      <div ref={containerRef} className="graph-viewer-container" style={{ flex: 1 }}>
                        <ForceGraph2D
                          ref={fgRef}
                          width={dimensions.width}
                          height={dimensions.height}
                          graphData={graphData}
                          nodeLabel={(node: any) => `${node.name} [${node.label}]`}
                          linkLabel={(link: any) => link.label}
                          nodeColor={(node: any) => node.color}
                          nodeVal={10}
                          linkDirectionalArrowLength={6}
                          linkDirectionalArrowRelPos={1}
                          linkWidth={2}
                          linkColor={() => '#2a2c3d'}
                          linkDirectionalArrowColor={() => '#3f425c'}
                          nodeCanvasObject={(node: any, ctx, globalScale) => {
                            const label = node.name;
                            const fontSize = 12 / globalScale;
                            ctx.font = `${fontSize}px Sans-Serif`;
                            const textWidth = ctx.measureText(label).width;
                            const bckgDimensions = [textWidth, fontSize].map(n => n + fontSize * 0.4);

                            ctx.fillStyle = 'rgba(10, 11, 16, 0.8)';
                            ctx.fillRect(node.x - bckgDimensions[0] / 2, node.y - bckgDimensions[1] / 2, ...bckgDimensions as [number, number]);

                            ctx.strokeStyle = node.color;
                            ctx.lineWidth = 1;
                            ctx.strokeRect(node.x - bckgDimensions[0] / 2, node.y - bckgDimensions[1] / 2, ...bckgDimensions as [number, number]);

                            ctx.textAlign = 'center';
                            ctx.textBaseline = 'middle';
                            ctx.fillStyle = node.color;
                            ctx.fillText(label, node.x, node.y);
                          }}
                        />
                        
                        <div className="graph-legend">
                          <div className="legend-item"><span className="legend-color" style={{ background: '#ec4899' }} /> Asset</div>
                          <div className="legend-item"><span className="legend-color" style={{ background: '#f97316' }} /> Component</div>
                          <div className="legend-item"><span className="legend-color" style={{ background: '#8b5cf6' }} /> Vendor</div>
                          <div className="legend-item"><span className="legend-color" style={{ background: '#3b82f6' }} /> Location</div>
                          <div className="legend-item"><span className="legend-color" style={{ background: '#ef4444' }} /> Incident</div>
                          <div className="legend-item"><span className="legend-color" style={{ background: '#14b8a6' }} /> Employee</div>
                          <div className="legend-item"><span className="legend-color" style={{ background: '#06b6d4' }} /> Team</div>
                        </div>
                      </div>
                    </>
                  )}
                </div>
              )}
            </>
          )}
        </div>
      </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
