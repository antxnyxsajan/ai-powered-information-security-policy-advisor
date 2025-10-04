import { useState, useEffect, useRef } from 'react';
import './App.css';

// --- NEW: Component to parse and render bold markdown ---
function FormattedMessage({ text }) {
  // This function finds **text** and replaces it with <strong>text</strong>
  const formatText = (rawText) => {
    // Also handles potential bullet points that start with * or -
    let formatted = rawText.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    formatted = formatted.replace(/^\s*[\*-]\s/gm, '<br>• ');
    return formatted;
  };

  return (
    <div
      dangerouslySetInnerHTML={{ __html: formatText(text) }}
    />
  );
}


const UserAvatar = () => (
  <div className="avatar">
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
      <path fillRule="evenodd" d="M7.5 6a4.5 4.5 0 119 0 4.5 4.5 0 01-9 0zM3.751 20.105a8.25 8.25 0 0116.498 0 .75.75 0 01-.437.695A18.683 18.683 0 0112 22.5c-2.786 0-5.433-.608-7.812-1.7a.75.75 0 01-.437-.695z" clipRule="evenodd" />
    </svg>
  </div>
);

const BotAvatar = () => (
  <div className="avatar">
    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" d="M8.625 9.75a.375.375 0 1 1-.75 0 .375.375 0 0 1 .75 0Zm0 0H8.25m4.125 0a.375.375 0 1 1-.75 0 .375.375 0 0 1 .75 0Zm0 0H12m4.125 0a.375.375 0 1 1-.75 0 .375.375 0 0 1 .75 0Zm0 0h-.375m-13.5 3.01c0 1.6 1.123 2.994 2.707 3.227 1.087.16 2.185.283 3.293.369V21l4.184-4.183a1.14 1.14 0 0 1 .778-.332 48.294 48.294 0 0 0 5.83-.498c1.585-.233 2.708-1.626 2.708-3.228V6.741c0-1.602-1.123-2.995-2.707-3.228A48.394 48.394 0 0 0 12 3c-2.392 0-4.744.175-7.043.513C3.373 3.746 2.25 5.14 2.25 6.741v6.018Z" />
    </svg>
  </div>
);

const SendIcon = () => (
  // Corrected fill color to be visible on the blue button
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="white" width="24px" height="24px">
    <path d="M3.478 2.405a.75.75 0 00-.926.94l2.432 7.905H13.5a.75.75 0 010 1.5H4.984l-2.432 7.905a.75.75 0 00.926.94 60.519 60.519 0 0018.445-8.986.75.75 0 000-1.218A60.517 60.517 0 003.478 2.405z" />
  </svg>
);

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const chatWindowRef = useRef(null);
  const API_URL = "http://127.0.0.1:8000/chat";

  useEffect(() => {
    if (chatWindowRef.current) {
      chatWindowRef.current.scrollTop = chatWindowRef.current.scrollHeight;
    }
  }, [messages, isLoading]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = { text: input, sender: 'user' };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: input })
      });

      if (!response.ok) throw new Error("Network response was not ok.");

      const data = await response.json();
      const botMessage = { text: data.answer, sender: 'bot' };
      setMessages(prev => [...prev, botMessage]);

    } catch (error) {
      console.error("Fetch error:", error);
      const errorMessage = { text: "Sorry, I'm having trouble connecting. Please try again.", sender: 'bot' };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="chat-container">
      <header className="chat-header">
        <h1>AI Security Advisor</h1>
      </header>
      <section className="chat-window" ref={chatWindowRef}>
        {messages.map((msg, index) => (
          <div key={index} className={`message-row ${msg.sender}`}>
            {msg.sender === 'bot' ? (
              <>
                <BotAvatar />
                <div className="message-bubble">
                  {/* --- CHANGE: Use the FormattedMessage component for bot messages --- */}
                  <FormattedMessage text={msg.text} />
                </div>
              </>
            ) : (
              <>
                <div className="message-bubble">{msg.text}</div>
                <UserAvatar />
              </>
            )}
          </div>
        ))}
        {isLoading && (
          <div className="message-row bot">
            <BotAvatar />
            <div className="message-bubble typing-indicator">
              <div className="dot"></div>
              <div className="dot"></div>
              <div className="dot"></div>
            </div>
          </div>
        )}
      </section>
      <form className="chat-form" onSubmit={handleSubmit}>
        <input
          type="text"
          id="chat-input"
          placeholder="Ask about company policies..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          autoComplete="off"
        />
        <button id="send-btn" type="submit" disabled={isLoading}>
          <SendIcon />
        </button>
      </form>
    </main>
  );
}

export default App;