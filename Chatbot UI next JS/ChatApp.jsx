import React, { useState, useRef, useEffect } from 'react';
import { MessageCircle, PlusCircle, Trash2, Send, Book, ExternalLink, ChevronRight, Clock, Rocket, User, Settings, MessageSquare } from 'lucide-react';
import { Sidebar } from 'primereact/sidebar'; // Import Sidebar

// Assuming you have PrimeReact CSS imported globally as mentioned in prerequisites

const IntegratedChatApp = () => {
  // State for chat messages
  const [messages, setMessages] = useState([
    { id: 1, text: "Hello! I'm your research assistant. How can I help you today?", sender: "bot", timestamp: "10:03 AM" },
    { id: 2, text: "Hi there! Can you tell me about P9 process?", sender: "user", timestamp: "10:04 AM" },
    { id: 3, text: "The P9 process from ProcessStandard is a structured approach to process management and improvement, particularly in the context of Automotive SPICE (Software Process Improvement and Capability dEtermination). It involves several key practices and attributes aimed at ensuring effective process deployment and performance.[1]", sender: "bot", timestamp: "10:04 AM" }
  ]);

  // State for the input field
  const [inputText, setInputText] = useState("");

  // State for the currently selected citation to display details
  const [activeCitation, setActiveCitation] = useState({
    id: 1,
    title: "Automotive_SPICE_PAM_30",
    source: "This is a document stating all the process standards for PAM 30",
    date: "March 2024",
    preview: "The P9 process from ProcessStandard is a structured approach to process management and improvement, particularly in the context of Automotive SPICE (Software Process Improvement and Capability dEtermination). It involves several key practices and attributes aimed at ensuring effective process deployment and performance.",
    url: "https://example.com/automotive-spice-pam-30",
    relevance: "High"
  });

  // Static list of citations (could be fetched dynamically)
  const [citations] = useState([
    {
      id: 1,
      title: "Automotive_SPICE_PAM_30",
      source: "This is a document stating all the process standards for PAM 30",
      date: "March 2024",
      preview: "The P9 process from ProcessStandard is a structured approach to process management and improvement, particularly in the context of Automotive SPICE (Software Process Improvement and Capability dEtermination). It involves several key practices and attributes aimed at ensuring effective process deployment and performance.",
      url: "https://example.com/automotive-spice-pam-30",
      relevance: "High"
    },
    {
      id: 2,
      title: "Automotive SIG_PAM_v25_changes",
      source: "This is a document stating all the process standards for PAM 25",
      date: "January 2024",
      preview: "This document outlines the changes between PAM v2.5 and PAM v3.0, which may indirectly relate to the P9 process understanding.",
      url: "https://example.com/automotive-sig-pam-v25-changes",
      relevance: "Medium"
    }
  ]);

  // Static list of chat history items (could be fetched dynamically)
  const [chatHistory] = useState([
    { id: 1, title: "in which quarter does the Aufgabenname EO: Project definerrt begin", date: "Today" },
    { id: 2, title: "Give me an example of identidfication and evaluation of hazardou use cases", date: "Yesterday" },
    { id: 3, title: "Overview of retest graph", date: "Feb 24" }
  ]);

  // State to control the visibility of header popovers
  const [showHeaderPopover, setShowHeaderPopover] = useState(null); // null, 'clock', 'rocket', 'profile'

  // --- New State for Knowledge Base Sidebar ---
  const [knowledgeBaseVisible, setKnowledgeBaseVisible] = useState(false);
  // ------------------------------------------

  // Ref to scroll to the bottom of the messages list
  const messagesEndRef = useRef(null);

  // Function to scroll the chat window to the bottom
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  // Effect to scroll down when new messages are added
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Function to handle sending a new message
  const handleSendMessage = () => {
    if (inputText.trim() === "") return; // Don't send empty messages

    const currentTime = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const newUserMessage = {
      id: messages.length + 1,
      text: inputText,
      sender: "user",
      timestamp: currentTime
    };

    // Add user message to state
    setMessages(prevMessages => [...prevMessages, newUserMessage]);
    setInputText(""); // Clear input field

    // Simulate a bot response after a short delay
    setTimeout(() => {
      const botTime = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
       const botMessageId = messages.length + 2; // Calculate ID based on potentially updated length
      const newBotMessage = {
        id: botMessageId, // Use calculated ID
        text: `Okay, regarding "${inputText}", here's some more information based on available documents. The P9 process specifically focuses on... [simulated response referencing citation 2] [${citations[1].id}]`, // Reference the citation ID
        sender: "bot",
        timestamp: botTime
      };
      // Add bot message and update the active citation based on the simulated response
      setMessages(prevMessages => [...prevMessages, newBotMessage]);
       // Find the citation referenced in the bot message (simple example)
      const mentionedCitationIdStr = newBotMessage.text.match(/\[(\d+)\]$/);
      if (mentionedCitationIdStr) {
        const mentionedCitationId = parseInt(mentionedCitationIdStr[1], 10);
        const mentionedCitation = citations.find(c => c.id === mentionedCitationId);
        if (mentionedCitation) {
          setActiveCitation(mentionedCitation);
        } else {
           // Fallback if citation not found, maybe activate the first one or null
           setActiveCitation(citations[0] || null);
        }
      } else {
        // Fallback if no citation mentioned, maybe activate the first one or null
        setActiveCitation(citations[0] || null);
      }
    }, 1000);
  };

  // Function to handle Enter key press in the input field
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { // Send on Enter, allow Shift+Enter for newline
      e.preventDefault(); // Prevent default newline behavior
      handleSendMessage();
    }
  };

  // Function to clear the current chat messages
  const handleClearChat = () => {
    const currentTime = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    setMessages([
      { id: 1, text: "Chat cleared. How can I help you today?", sender: "bot", timestamp: currentTime }
    ]);
    setActiveCitation(null); // Clear active citation when chat is cleared
  };

  // Function to start a new chat session
  const handleNewChat = () => {
    const currentTime = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    setMessages([
      { id: 1, text: "Started a new chat. What would you like to discuss?", sender: "bot", timestamp: currentTime }
    ]);
    setActiveCitation(null); // Clear active citation for a new chat
  };

  // Function to set the active citation when a citation item is clicked
  const selectCitation = (citation) => {
    setActiveCitation(citation);
  };

  // Function to toggle the visibility of header popovers
  const toggleHeaderPopover = (popoverId) => {
    setShowHeaderPopover(currentPopover => currentPopover === popoverId ? null : popoverId);
  };

  return (
    <div className="flex flex-col h-screen bg-gray-50 font-sans antialiased">
      {/* Integrated Header (unchanged) */}
      <div className="w-full shadow-md z-20" style={{ backgroundColor: "#002733" }}>
        <div className="container mx-auto px-4 py-2">
          <div className="flex items-center justify-between">
            {/* Left side of Header */}
            <div className="flex items-center">
              <h5 className="font-bold text-lg mr-8 text-white">PMT Assistant</h5>
            </div>
            {/* Right side of Header (unchanged) */}
            <div className="flex items-center">
              <div className="h-6 w-px bg-gray-600 mx-4"></div>
              <div className="flex items-center space-x-4">
                 {/* Clock Icon */}
                 <div className="relative">
                   <button
                     onClick={() => toggleHeaderPopover('clock')}
                     className="text-gray-300 hover:text-white transition-colors duration-200 focus:outline-none"
                     aria-label="View recent activity"
                   >
                     <Clock size={22} />
                   </button>
                   {showHeaderPopover === 'clock' && (
                     <div className="absolute right-0 mt-3 w-64 bg-white rounded-lg shadow-xl p-4 z-30 border border-gray-200">
                       <p className="text-sm font-medium text-gray-800 mb-2">Recent Activity</p>
                       <p className="text-xs text-gray-600">
                         No new notifications. Your recent chats and actions will appear here.
                       </p>
                     </div>
                   )}
                 </div>
                 {/* Rocket Icon */}
                 <div className="relative">
                   <button
                     onClick={() => toggleHeaderPopover('rocket')}
                     className="text-gray-300 hover:text-white transition-colors duration-200 focus:outline-none"
                     aria-label="Quick actions"
                   >
                     <Rocket size={22} />
                   </button>
                   {showHeaderPopover === 'rocket' && (
                     <div className="absolute right-0 mt-3 w-64 bg-white rounded-lg shadow-xl p-4 z-30 border border-gray-200">
                        <p className="text-sm font-medium text-gray-800 mb-2">Quick Actions</p>
                        <ul className="text-xs text-gray-600 space-y-1">
                          <li>- Start New Analysis</li>
                          <li>- Generate Report</li>
                          <li>- Access Knowledge Base</li>
                        </ul>
                     </div>
                   )}
                 </div>
                 {/* User Profile Icon */}
                 <div className="relative">
                   <button
                     onClick={() => toggleHeaderPopover('profile')}
                     className="rounded-full h-8 w-8 flex items-center justify-center text-white hover:opacity-90 transition-opacity duration-200 focus:outline-none ring-2 ring-offset-2 ring-offset-[#002733]"
                     style={{ backgroundColor: "#008075", ringColor: "#c2fe06" }}
                     aria-label="User profile and settings"
                   >
                     <User size={18} />
                   </button>
                   {showHeaderPopover === 'profile' && (
                     <div className="absolute right-0 mt-3 w-64 bg-white rounded-lg shadow-xl p-4 z-30 border border-gray-200">
                       <p className="text-sm font-medium text-gray-800 mb-2">User Profile</p>
                       <p className="text-xs text-gray-600 mb-3">example.user@domain.com</p>
                       <button className="text-xs text-[#008075] hover:underline">Account Settings</button>
                       <div className="border-t border-gray-200 my-2"></div>
                       <button className="text-xs text-red-600 hover:underline">Logout</button>
                     </div>
                   )}
                 </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Chat Application Body */}
      <div className="flex flex-1 overflow-hidden">

        {/* Left Sidebar */}
        <div className="w-1/5 max-w-xs bg-white border-r border-gray-200 flex flex-col flex-shrink-0">
          <div className="flex flex-1 overflow-hidden">
            {/* First Panel: Icons */}
            <div className="w-16 border-r border-gray-200 flex flex-col items-center py-4 space-y-4 flex-shrink-0">
              {/* --- Updated Knowledge Base Button --- */}
              <button
                onClick={() => setKnowledgeBaseVisible(true)} // Set state to show sidebar
                className="p-2 rounded-lg text-gray-600 hover:bg-gray-100 hover:text-gray-900 transition-colors duration-150"
                title="Knowledge Base"
              >
                <Book size={20} />
              </button>
              {/* -------------------------------------- */}
              <button className="p-2 rounded-lg text-gray-600 hover:bg-gray-100 hover:text-gray-900 transition-colors duration-150" title="Settings">
                <Settings size={20} />
              </button>
              <button className="p-2 rounded-lg text-gray-600 hover:bg-gray-100 hover:text-gray-900 transition-colors duration-150" title="Feedback">
                <MessageSquare size={20} />
              </button>
            </div>

            {/* Second Panel: New Chat + Chat History (unchanged) */}
            <div className="flex-1 flex flex-col min-w-0">
              {/* New Chat Button */}
              <div className="p-3 border-b border-gray-200 flex-shrink-0">
                <button
                  onClick={handleNewChat}
                  className="w-full text-white font-medium py-2 px-4 rounded-lg flex items-center justify-center gap-2 shadow-sm transition-all duration-200 hover:opacity-90 focus:outline-none focus:ring-2 focus:ring-offset-2"
                  style={{ backgroundColor: "#008075", focusRingColor: "#c2fe06" }}
                >
                  <PlusCircle size={18} />
                  <span>New Chat</span>
                </button>
              </div>
              {/* Chat History List */}
              <div className="flex-1 p-3 overflow-y-auto">
                <h3 className="text-xs uppercase text-gray-500 font-semibold mb-3 tracking-wider px-1">Recent Chats</h3>
                <div className="space-y-1">
                  {chatHistory.map(chat => (
                    <div
                      key={chat.id}
                      className="p-2 rounded-lg hover:bg-gray-100 cursor-pointer transition-colors duration-150 group"
                      role="button"
                      tabIndex={0}
                    >
                      <div className="flex items-center gap-2.5">
                        <MessageCircle size={16} className="flex-shrink-0" style={{ color: "#008075" }} />
                        <div className="flex-1 overflow-hidden">
                          <p className="text-sm font-medium text-gray-800 truncate" title={chat.title}>{chat.title}</p>
                          <p className="text-xs text-gray-500">{chat.date}</p>
                        </div>
                        <ChevronRight size={16} className="text-gray-400 opacity-0 group-hover:opacity-100 transition-opacity duration-150 ml-auto flex-shrink-0" />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
          {/* Clear Chat Button (unchanged) */}
          <div className="p-4 border-t border-gray-200 flex-shrink-0">
              <button
                  onClick={handleClearChat}
                  className="text-gray-700 hover:bg-gray-100 font-medium py-2 px-4 rounded-lg w-full flex items-center justify-center gap-2 transition-all duration-200 border border-gray-200 hover:border-gray-300"
              >
                  <Trash2 size={18} />
                  <span>Clear Chat</span>
              </button>
          </div>
        </div>

        {/* Middle Chat Section (minor update for bot response) */}
        <div className="w-2/5 flex flex-col bg-white border-r border-gray-200">
          <div className="p-4 border-b border-gray-200 flex justify-between items-center flex-shrink-0" style={{ backgroundColor: "#002733", color: "white" }}>
            <h2 className="font-semibold text-lg">Research Chat</h2>
            <div className="text-sm opacity-80">
              {messages.length} messages
            </div>
          </div>

          <div className="flex-1 p-4 overflow-y-auto bg-gray-50">
            {messages.map(message => (
              <div
                key={message.id}
                className={`mb-6 flex ${ message.sender === "user" ? "justify-end" : "justify-start" }`}
              >
                <div
                  className={`max-w-xs md:max-w-md lg:max-w-lg rounded-t-2xl ${ message.sender === "user" ? "text-white rounded-bl-2xl" : "bg-white text-gray-800 rounded-br-2xl border border-gray-200 shadow-sm" }`}
                  style={message.sender === "user" ? { backgroundColor: "#008075" } : {}}
                >
                  <div className="p-4">
                    {/* Render text, potentially handling citation links */}
                    {message.sender === 'bot' ? (
                      <span>
                        {message.text.replace(/\[(\d+)\]/g, (match, id) => {
                          const citation = citations.find(c => c.id === parseInt(id, 10));
                          return citation ? (
                            <button
                              key={citation.id}
                              onClick={() => selectCitation(citation)}
                              className="inline-block align-middle mx-0.5 px-1.5 py-0.5 text-xs font-mono rounded text-white cursor-pointer hover:opacity-80"
                              style={{ backgroundColor: "#008075" }}
                              title={`View citation: ${citation.title}`}
                            >
                              {id}
                            </button>
                          ) : (
                            match // Keep original text if citation not found
                          );
                        })}
                      </span>
                    ) : (
                      message.text // Render user text normally
                    )}
                  </div>
                  <div
                    className={`px-4 py-2 text-xs ${ message.sender === "user" ? "text-gray-100" : "text-gray-500" } flex justify-between items-center border-t ${ message.sender === "user" ? "" : "border-gray-100" }`}
                    style={message.sender === "user" ? { borderColor: "rgba(255,255,255,0.2)" } : {}}
                  >
                    <span>
                      {message.sender === "user" ? "You" : "Assistant"}
                    </span>
                    <span>{message.timestamp}</span>
                  </div>
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>

          <div className="p-4 border-t border-gray-200 bg-white flex-shrink-0">
            <div className="flex items-center bg-gray-50 rounded-lg border border-gray-200 focus-within:ring-2 transition-all duration-200 focus-within:border-[#008075] focus-within:ring-[#c2fe06]/30">
               <input
                 type="text"
                 value={inputText}
                 onChange={(e) => setInputText(e.target.value)}
                 onKeyPress={handleKeyPress}
                 placeholder="Type your research question..."
                 className="flex-1 p-3 bg-transparent border-none focus:outline-none text-gray-800 placeholder-gray-500"
               />
               <button
                 onClick={handleSendMessage}
                 disabled={inputText.trim() === ""}
                 className={`m-1 p-2 rounded-md text-white transition-colors duration-200 ${inputText.trim() === "" ? 'bg-gray-400 cursor-not-allowed' : 'bg-[#008075] hover:bg-[#00665e]'}`}
                 aria-label="Send message"
               >
                 <Send size={18} />
               </button>
             </div>
          </div>
        </div>

        {/* Right Citation Preview Section (unchanged structure, updated styles) */}
        <div className="w-2/5 bg-white flex flex-col">
            <div className="p-4 border-b border-gray-200 flex items-center gap-2 flex-shrink-0" style={{ backgroundColor: "#002733", color: "white" }}>
                <Book size={20} style={{ color: "#c2fe06" }} />
                <h2 className="font-semibold text-lg">Citations</h2>
            </div>

            <div className="flex flex-col flex-1 overflow-hidden"> {/* Allow sections inside to scroll */}
                {/* Referenced Sources List */}
                <div className="flex-none p-4 border-b border-gray-200 overflow-y-auto max-h-48"> {/* Limit height and allow scroll */}
                    <h3 className="text-xs uppercase text-gray-500 font-semibold mb-2 tracking-wider">Referenced Sources</h3>
                    {citations.map(citation => (
                        <div
                            key={citation.id}
                            onClick={() => selectCitation(citation)}
                            className={`p-3 mb-2 rounded-lg cursor-pointer transition-all duration-150 flex gap-3 items-center ${ activeCitation?.id === citation.id ? "border bg-[#e6f2f1]" : "hover:bg-gray-50 border border-transparent" }`}
                            style={activeCitation?.id === citation.id ? { borderColor: "#008075" } : {}}
                            role="button"
                            tabIndex={0}
                        >
                            <div className="font-mono text-sm h-6 w-6 rounded-full flex items-center justify-center flex-shrink-0 text-white"
                                style={{ backgroundColor: "#008075" }}>
                                {citation.id}
                            </div>
                            <div className="flex-1 overflow-hidden">
                                <p className="font-medium text-gray-900 text-sm truncate" title={citation.title}>{citation.title}</p>
                                <p className="text-xs text-gray-500 truncate">{citation.source} • {citation.date}</p>
                            </div>
                        </div>
                    ))}
                </div>

                {/* Active Citation Detail View */}
                <div className="flex-1 p-4 overflow-y-auto"> {/* Allow details to scroll */}
                    {activeCitation ? (
                        <div className="h-full flex flex-col">
                            <div className="bg-gray-50 rounded-xl p-5 mb-4 border border-gray-200 flex-grow"> {/* Use flex-grow to take available space */}
                                <div className="flex justify-between items-start mb-2 gap-2">
                                    <h3 className="font-bold text-lg text-gray-900">{activeCitation.title}</h3>
                                     <span className={`text-xs px-2 py-0.5 rounded-full font-medium whitespace-nowrap ${activeCitation.relevance === "High" ? 'bg-[#e0ffb3] text-[#3d521a]' : 'bg-gray-200 text-gray-700'}`}>
                                        {activeCitation.relevance} Relevance
                                    </span>
                                </div>

                                <div className="flex flex-wrap gap-x-2 items-center mb-4 text-sm">
                                    <p className="font-medium text-gray-700">{activeCitation.source}</p>
                                    <span className="text-gray-400">•</span>
                                    <p className="text-gray-600">{activeCitation.date}</p>
                                </div>

                                <div className="prose prose-sm max-w-none text-gray-700 mb-6">
                                    <p>{activeCitation.preview}</p>
                                </div>

                                <a
                                    href={activeCitation.url}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="text-sm font-medium flex items-center gap-1 transition-colors duration-150 hover:underline"
                                    style={{ color: "#008075" }}
                                >
                                    View original source <ExternalLink size={14} className="inline-block"/>
                                </a>
                            </div>
                            {/* Optional bottom section, removed placeholder */}
                        </div>
                    ) : (
                        <div className="h-full flex items-center justify-center text-center text-gray-500">
                             <div className="p-8">
                                 <div className="rounded-full p-4 bg-[#e6f2f1] inline-flex items-center justify-center mb-4">
                                     <Book size={24} style={{ color: "#008075" }} />
                                 </div>
                                 <h3 className="text-lg font-medium text-gray-700 mb-2">No citation selected</h3>
                                 <p className="text-sm text-gray-500 max-w-xs mx-auto">
                                     Click on a citation number [1] in the chat or select one from the list above to view details.
                                 </p>
                             </div>
                         </div>
                    )}
                </div>
            </div>
        </div>

      </div>

      {/* --- Knowledge Base Sidebar Definition --- */}
      <Sidebar
        visible={knowledgeBaseVisible}
        onHide={() => setKnowledgeBaseVisible(false)}
        fullScreen
        className="p-4" // Add some padding inside the sidebar
      >
        <div className="flex justify-between items-center mb-4 pb-2 border-b border-gray-300">
            <h2 className="text-2xl font-semibold text-gray-800">Knowledge Base</h2>
             {/* You might want a styled close button if the default isn't prominent enough */}
             {/* <button onClick={() => setKnowledgeBaseVisible(false)} className="...">Close</button> */}
        </div>

        {/* Add your Knowledge Base content here */}
        <p className="text-gray-700">
          Welcome to the Knowledge Base. Here you can find documents, articles, and other resources.
        </p>
        <p className="mt-4 text-gray-700">
          Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
          Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
        </p>
        {/* Example: Add a search bar or list of documents */}
        <div className="mt-6">
          <input type="search" placeholder="Search knowledge base..." className="w-full p-2 border border-gray-300 rounded-md"/>
        </div>
      </Sidebar>
      {/* ---------------------------------------- */}

    </div>
  );
};

export default IntegratedChatApp;
