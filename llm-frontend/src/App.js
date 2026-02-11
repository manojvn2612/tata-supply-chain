

import React, { useState, useRef, useEffect } from 'react';
import './App.css';

function mockLLMResponse(message) {
  // Simulate a response from an LLM
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve("Sure! " + message.split('').reverse().join(''));
    }, 800);
  });
}

const AVATARS = {
  user: 'https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/72x72/1f464.png',
  llm: 'https://cdn-icons-png.flaticon.com/512/4712/4712035.png', // new robot image
};

function App() {
  const [messages, setMessages] = useState([
    { sender: 'llm', text: 'Hello! How can I help you today?' }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const chatEndRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;
    const userMsg = { sender: 'user', text: input };
    setMessages((msgs) => [...msgs, userMsg]);
    setInput('');
    setLoading(true);
    const llmReply = await mockLLMResponse(input);
    setMessages((msgs) => [...msgs, { sender: 'llm', text: llmReply }]);
    setLoading(false);
  };

  return (
    <div className="gpt-root">
      <div className="gpt-header">Supply Chain Optimizer</div>
      <div className="gpt-chat-window">
        {messages.map((msg, idx) => (
          <div key={idx} className={`gpt-message-row ${msg.sender}`}>
            <img className="gpt-avatar" src={AVATARS[msg.sender]} alt={msg.sender} />
            <div className={`gpt-bubble ${msg.sender}`}>{msg.text}</div>
          </div>
        ))}
        {loading && (
          <div className="gpt-message-row llm">
            <img className="gpt-avatar" src={AVATARS.llm} alt="llm" />
            <div className="gpt-bubble llm loading">LLM is typing...</div>
          </div>
        )}
        <div ref={chatEndRef} />
      </div>
      <form className="gpt-input-bar" onSubmit={handleSend}>
        <input
          className="gpt-input"
          type="text"
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder="Message LLM..."
          disabled={loading}
          autoFocus
        />
        <button className="gpt-send-btn" type="submit" disabled={loading || !input.trim()}>
          <svg width="24" height="24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>
        </button>
      </form>
    </div>
  );
}

export default App;
