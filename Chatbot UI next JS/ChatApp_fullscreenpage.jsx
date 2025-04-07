import React, { useState, useRef, useEffect } from 'react';
import { 
    MessageCircle, PlusCircle, Trash2, Send, Book, ExternalLink, ChevronRight, 
    Clock, Rocket, User, Settings, MessageSquare 
} from 'lucide-react';
import { Sidebar } from 'primereact/sidebar';
import { Button } from 'primereact/button';
import { DataTable } from 'primereact/datatable';
import { Column } from 'primereact/column';

/*
 * Make sure PrimeReact CSS is imported in your project's entry point (e.g., index.js or App.js):
 * import 'primereact/resources/themes/saga-blue/theme.css'; // or your chosen theme
 * import 'primereact/resources/primereact.min.css';
 * import 'primeicons/primeicons.css';
 */

const IntegratedChatApp = () => {
    // --- State Variables ---

    // Chat Messages
    const [messages, setMessages] = useState([
        { id: 1, text: "Hello! I'm your research assistant. How can I help you today?", sender: "bot", timestamp: "10:03 AM" },
        { id: 2, text: "Hi there! Can you tell me about P9 process?", sender: "user", timestamp: "10:04 AM" },
        { id: 3, text: "The P9 process from ProcessStandard is a structured approach to process management and improvement, particularly in the context of Automotive SPICE (Software Process Improvement and Capability dEtermination). It involves several key practices and attributes aimed at ensuring effective process deployment and performance.[1]", sender: "bot", timestamp: "10:04 AM" }
    ]);

    // Input Field
    const [inputText, setInputText] = useState("");

    // Citations
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
     const [activeCitation, setActiveCitation] = useState(citations[0]); // Start with the first citation active

    // Chat History (Static Example)
    const [chatHistory] = useState([
        { id: 1, title: "in which quarter does the Aufgabenname EO: Project definerrt begin", date: "Today" },
        { id: 2, title: "Give me an example of identidfication and evaluation of hazardou use cases", date: "Yesterday" },
        { id: 3, title: "Overview of retest graph", date: "Feb 24" }
    ]);

    // Header Popovers
    const [showHeaderPopover, setShowHeaderPopover] = useState(null); // null, 'clock', 'rocket', 'profile'

    // Knowledge Base Sidebar Visibility
    const [knowledgeBaseVisible, setKnowledgeBaseVisible] = useState(false);

    // Knowledge Base Documents Data
    const [knowledgeBaseDocs, setKnowledgeBaseDocs] = useState([
        { id: 1, srNo: 1, fileName: 'Requirements_Spec_v1.2.pdf', lastModified: '2024-03-20 11:45', size: '1.8 MB' },
        { id: 2, srNo: 2, fileName: 'Design_Document_RevA.docx', lastModified: '2024-03-18 09:00', size: '3.2 MB' },
        { id: 3, srNo: 3, fileName: 'User_Manual_Draft.pdf', lastModified: '2024-03-21 16:10', size: '950 KB' },
        { id: 4, srNo: 4, fileName: 'API_Reference.html', lastModified: '2024-03-15 10:30', size: '500 KB' },
        // Add more sample documents
    ]);

    // --- Refs ---
    const messagesEndRef = useRef(null);

    // --- Effects ---
    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    // --- Helper Functions ---

    // Scroll chat to bottom
    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    // Send message handler
    const handleSendMessage = () => {
        if (inputText.trim() === "") return;

        const currentTime = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        const newUserMessage = {
            id: Date.now(), // Use timestamp for unique ID
            text: inputText,
            sender: "user",
            timestamp: currentTime
        };

        setMessages(prevMessages => [...prevMessages, newUserMessage]);
        setInputText("");

        // Simulate bot response
        setTimeout(() => {
            const botTime = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            // Example: Bot references the second citation
            const botResponseCitationId = 2; 
            const newBotMessage = {
                id: Date.now() + 1, // Unique ID
                text: `Okay, regarding "${inputText}", the relevant details can often be found in documents like [${botResponseCitationId}]. The P9 process specifically focuses on... [simulated response]`,
                sender: "bot",
                timestamp: botTime
            };

            setMessages(prevMessages => [...prevMessages, newBotMessage]);

            // Activate the citation mentioned by the bot
            const mentionedCitation = citations.find(c => c.id === botResponseCitationId);
            if (mentionedCitation) {
                setActiveCitation(mentionedCitation);
            }

        }, 1000);
    };

    // Handle Enter key press
    const handleKeyPress = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSendMessage();
        }
    };

    // Clear chat handler
    const handleClearChat = () => {
        const currentTime = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        setMessages([
            { id: Date.now(), text: "Chat cleared. How can I help you today?", sender: "bot", timestamp: currentTime }
        ]);
        setActiveCitation(null);
    };

    // New chat handler
    const handleNewChat = () => {
        const currentTime = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        setMessages([
            { id: Date.now(), text: "Started a new chat. What would you like to discuss?", sender: "bot", timestamp: currentTime }
        ]);
        setActiveCitation(null);
    };

    // Select citation handler
    const selectCitation = (citation) => {
        setActiveCitation(citation);
    };

    // Toggle header popover visibility
    const toggleHeaderPopover = (popoverId) => {
        setShowHeaderPopover(currentPopover => currentPopover === popoverId ? null : popoverId);
    };

    // --- Knowledge Base Action Handlers (Placeholders) ---
    const handleEditDocument = (doc) => {
        console.log("Edit document:", doc);
        alert(`Edit action for: ${doc.fileName}`);
        // Implement edit logic (e.g., open a modal)
    };

    const handleDeleteDocument = (doc) => {
        console.log("Delete document:", doc);
        if (window.confirm(`Are you sure you want to delete ${doc.fileName}?`)) {
            setKnowledgeBaseDocs(prevDocs => prevDocs.filter(d => d.id !== doc.id));
            console.log("Document deleted (from state)");
        }
    };
    
    const handleUploadDocument = () => {
        console.log("Upload document clicked");
        alert("Upload functionality to be implemented.");
    };

    // --- Table Body Templates ---

    // Template for Actions column in Knowledge Base table
    const documentActionBodyTemplate = (rowData) => {
        return (
            <div className="flex justify-center gap-2">
                <Button
                    icon="pi pi-pencil"
                    className="p-button-rounded p-button-success p-button-text p-button-sm"
                    onClick={() => handleEditDocument(rowData)}
                    tooltip="Edit"
                    tooltipOptions={{ position: 'top' }}
                />
                <Button
                    icon="pi pi-trash"
                    className="p-button-rounded p-button-danger p-button-text p-button-sm"
                    onClick={() => handleDeleteDocument(rowData)}
                    tooltip="Delete"
                    tooltipOptions={{ position: 'top' }}
                 />
            </div>
        );
    };

    // --- Render Component ---
    return (
        <div className="flex flex-col h-screen bg-gray-50 font-sans antialiased">
            
            {/* ======================= Header ======================= */}
            <div className="w-full shadow-md z-20" style={{ backgroundColor: "#002733" }}>
                <div className="container mx-auto px-4 py-2">
                    <div className="flex items-center justify-between">
                        {/* Left side */}
                        <div className="flex items-center">
                            <h5 className="font-bold text-lg mr-8 text-white">PMT Assistant</h5>
                        </div>
                        {/* Right side */}
                        <div className="flex items-center">
                            <div className="h-6 w-px bg-gray-600 mx-4"></div>
                            <div className="flex items-center space-x-4">
                                {/* Clock Icon */}
                                <div className="relative">
                                    <button onClick={() => toggleHeaderPopover('clock')} className="text-gray-300 hover:text-white transition-colors duration-200 focus:outline-none" aria-label="View recent activity"> <Clock size={22} /> </button>
                                    {showHeaderPopover === 'clock' && ( <div className="absolute right-0 mt-3 w-64 bg-white rounded-lg shadow-xl p-4 z-30 border border-gray-200"> <p className="text-sm font-medium text-gray-800 mb-2">Recent Activity</p> <p className="text-xs text-gray-600"> No new notifications. </p> </div> )}
                                </div>
                                {/* Rocket Icon */}
                                <div className="relative">
                                    <button onClick={() => toggleHeaderPopover('rocket')} className="text-gray-300 hover:text-white transition-colors duration-200 focus:outline-none" aria-label="Quick actions"> <Rocket size={22} /> </button>
                                    {showHeaderPopover === 'rocket' && ( <div className="absolute right-0 mt-3 w-64 bg-white rounded-lg shadow-xl p-4 z-30 border border-gray-200"> <p className="text-sm font-medium text-gray-800 mb-2">Quick Actions</p> <ul className="text-xs text-gray-600 space-y-1"> <li>- Start New Analysis</li> <li>- Generate Report</li> </ul> </div> )}
                                </div>
                                {/* User Profile Icon */}
                                <div className="relative">
                                    <button onClick={() => toggleHeaderPopover('profile')} className="rounded-full h-8 w-8 flex items-center justify-center text-white hover:opacity-90 transition-opacity duration-200 focus:outline-none ring-2 ring-offset-2 ring-offset-[#002733]" style={{ backgroundColor: "#008075", ringColor: "#c2fe06" }} aria-label="User profile and settings"> <User size={18} /> </button>
                                    {showHeaderPopover === 'profile' && ( <div className="absolute right-0 mt-3 w-64 bg-white rounded-lg shadow-xl p-4 z-30 border border-gray-200"> <p className="text-sm font-medium text-gray-800 mb-2">User Profile</p> <p className="text-xs text-gray-600 mb-3">example.user@domain.com</p> <button className="text-xs text-[#008075] hover:underline">Account Settings</button> <div className="border-t border-gray-200 my-2"></div> <button className="text-xs text-red-600 hover:underline">Logout</button> </div> )}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* ======================= Main Body ======================= */}
            <div className="flex flex-1 overflow-hidden">

                {/* ------------- Left Sidebar ------------- */}
                <div className="w-1/5 max-w-xs bg-white border-r border-gray-200 flex flex-col flex-shrink-0">
                    <div className="flex flex-1 overflow-hidden">
                        {/* Panel 1: Icons */}
                        <div className="w-16 border-r border-gray-200 flex flex-col items-center py-4 space-y-4 flex-shrink-0">
                            <button onClick={() => setKnowledgeBaseVisible(true)} className="p-2 rounded-lg text-gray-600 hover:bg-gray-100 hover:text-gray-900 transition-colors duration-150" title="Knowledge Base"> <Book size={20} /> </button>
                            <button className="p-2 rounded-lg text-gray-600 hover:bg-gray-100 hover:text-gray-900 transition-colors duration-150" title="Settings"> <Settings size={20} /> </button>
                            <button className="p-2 rounded-lg text-gray-600 hover:bg-gray-100 hover:text-gray-900 transition-colors duration-150" title="Feedback"> <MessageSquare size={20} /> </button>
                        </div>

                        {/* Panel 2: New Chat + History */}
                        <div className="flex-1 flex flex-col min-w-0">
                            {/* New Chat Button */}
                            <div className="p-3 border-b border-gray-200 flex-shrink-0">
                                <button onClick={handleNewChat} className="w-full text-white font-medium py-2 px-4 rounded-lg flex items-center justify-center gap-2 shadow-sm transition-all duration-200 hover:opacity-90 focus:outline-none focus:ring-2 focus:ring-offset-2" style={{ backgroundColor: "#008075", focusRingColor: "#c2fe06" }}> <PlusCircle size={18} /> <span>New Chat</span> </button>
                            </div>
                            {/* Chat History List */}
                            <div className="flex-1 p-3 overflow-y-auto">
                                <h3 className="text-xs uppercase text-gray-500 font-semibold mb-3 tracking-wider px-1">Recent Chats</h3>
                                <div className="space-y-1">
                                    {chatHistory.map(chat => (
                                        <div key={chat.id} className="p-2 rounded-lg hover:bg-gray-100 cursor-pointer transition-colors duration-150 group" role="button" tabIndex={0} /* onClick={() => loadChat(chat.id)} */ >
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
                    {/* Clear Chat Button */}
                    <div className="p-4 border-t border-gray-200 flex-shrink-0">
                        <button onClick={handleClearChat} className="text-gray-700 hover:bg-gray-100 font-medium py-2 px-4 rounded-lg w-full flex items-center justify-center gap-2 transition-all duration-200 border border-gray-200 hover:border-gray-300"> <Trash2 size={18} /> <span>Clear Chat</span> </button>
                    </div>
                </div>

                {/* ------------- Middle Chat Section ------------- */}
                <div className="w-2/5 flex flex-col bg-white border-r border-gray-200">
                    {/* Chat Header */}
                    <div className="p-4 border-b border-gray-200 flex justify-between items-center flex-shrink-0" style={{ backgroundColor: "#002733", color: "white" }}>
                        <h2 className="font-semibold text-lg">Research Chat</h2>
                        <div className="text-sm opacity-80"> {messages.length} messages </div>
                    </div>
                    {/* Message List */}
                    <div className="flex-1 p-4 overflow-y-auto bg-gray-50">
                        {messages.map(message => (
                            <div key={message.id} className={`mb-6 flex ${ message.sender === "user" ? "justify-end" : "justify-start" }`}>
                                <div className={`max-w-xs md:max-w-md lg:max-w-lg rounded-t-2xl ${ message.sender === "user" ? "text-white rounded-bl-2xl" : "bg-white text-gray-800 rounded-br-2xl border border-gray-200 shadow-sm" }`} style={message.sender === "user" ? { backgroundColor: "#008075" } : {}}>
                                    <div className="p-4">
                                        {/* Render text, linking citation numbers */}
                                        {message.sender === 'bot' ? (
                                            <span> {message.text.split(/(\[\d+\])/g).map((part, index) => {
                                                const match = part.match(/\[(\d+)\]/);
                                                if (match) {
                                                    const citationId = parseInt(match[1], 10);
                                                    const citation = citations.find(c => c.id === citationId);
                                                    return citation ? (
                                                        <button key={`${message.id}-cite-${index}`} onClick={() => selectCitation(citation)} className="inline-block align-middle mx-0.5 px-1.5 py-0.5 text-xs font-mono rounded text-white cursor-pointer hover:opacity-80 focus:outline-none focus:ring-1 focus:ring-white" style={{ backgroundColor: "#00665e" /* Darker teal */ }} title={`View citation: ${citation.title}`} > {citationId} </button>
                                                    ) : ( <span key={`${message.id}-text-${index}`}>{part}</span> ); // Render as text if citation not found
                                                }
                                                return <span key={`${message.id}-text-${index}`}>{part}</span>; // Render normal text part
                                            })} </span>
                                        ) : ( message.text /* Render user text normally */ )}
                                    </div>
                                    <div className={`px-4 py-2 text-xs ${ message.sender === "user" ? "text-gray-100" : "text-gray-500" } flex justify-between items-center border-t`} style={message.sender === "user" ? { borderColor: "rgba(255,255,255,0.2)" } : { borderColor: "rgba(0,0,0,0.05)" }}>
                                        <span>{message.sender === "user" ? "You" : "Assistant"}</span>
                                        <span>{message.timestamp}</span>
                                    </div>
                                </div>
                            </div>
                        ))}
                        <div ref={messagesEndRef} /> {/* Anchor for scrolling */}
                    </div>
                    {/* Input Area */}
                    <div className="p-4 border-t border-gray-200 bg-white flex-shrink-0">
                        <div className="flex items-center bg-gray-50 rounded-lg border border-gray-200 focus-within:ring-2 transition-all duration-200 focus-within:border-[#008075] focus-within:ring-[#c2fe06]/30">
                            <input type="text" value={inputText} onChange={(e) => setInputText(e.target.value)} onKeyPress={handleKeyPress} placeholder="Type your research question..." className="flex-1 p-3 bg-transparent border-none focus:outline-none text-gray-800 placeholder-gray-500" />
                            <button onClick={handleSendMessage} disabled={inputText.trim() === ""} className={`m-1 p-2 rounded-md text-white transition-colors duration-200 ${inputText.trim() === "" ? 'bg-gray-400 cursor-not-allowed' : 'bg-[#008075] hover:bg-[#00665e]'}`} aria-label="Send message"> <Send size={18} /> </button>
                        </div>
                    </div>
                </div>

                {/* ------------- Right Citation Preview Section ------------- */}
                <div className="w-2/5 bg-white flex flex-col">
                    {/* Citation Header */}
                    <div className="p-4 border-b border-gray-200 flex items-center gap-2 flex-shrink-0" style={{ backgroundColor: "#002733", color: "white" }}> <Book size={20} style={{ color: "#c2fe06" }} /> <h2 className="font-semibold text-lg">Citations</h2> </div>

                    <div className="flex flex-col flex-1 overflow-hidden">
                        {/* Referenced Sources List */}
                        <div className="flex-none p-4 border-b border-gray-200 overflow-y-auto max-h-48">
                            <h3 className="text-xs uppercase text-gray-500 font-semibold mb-2 tracking-wider">Referenced Sources</h3>
                            {citations.map(citation => (
                                <div key={citation.id} onClick={() => selectCitation(citation)} className={`p-3 mb-2 rounded-lg cursor-pointer transition-all duration-150 flex gap-3 items-center ${ activeCitation?.id === citation.id ? "border bg-[#e6f2f1]" : "hover:bg-gray-50 border border-transparent" }`} style={activeCitation?.id === citation.id ? { borderColor: "#008075" } : {}} role="button" tabIndex={0} >
                                    <div className="font-mono text-sm h-6 w-6 rounded-full flex items-center justify-center flex-shrink-0 text-white" style={{ backgroundColor: "#008075" }}> {citation.id} </div>
                                    <div className="flex-1 overflow-hidden">
                                        <p className="font-medium text-gray-900 text-sm truncate" title={citation.title}>{citation.title}</p>
                                        <p className="text-xs text-gray-500 truncate">{citation.source} • {citation.date}</p>
                                    </div>
                                </div>
                            ))}
                        </div>

                        {/* Active Citation Detail View */}
                        <div className="flex-1 p-4 overflow-y-auto">
                            {activeCitation ? (
                                <div className="h-full flex flex-col">
                                    <div className="bg-gray-50 rounded-xl p-5 mb-4 border border-gray-200 flex-grow">
                                        <div className="flex justify-between items-start mb-2 gap-2">
                                            <h3 className="font-bold text-lg text-gray-900">{activeCitation.title}</h3>
                                            <span className={`text-xs px-2 py-0.5 rounded-full font-medium whitespace-nowrap ${activeCitation.relevance === "High" ? 'bg-[#e0ffb3] text-[#3d521a]' : 'bg-gray-200 text-gray-700'}`}> {activeCitation.relevance} Relevance </span>
                                        </div>
                                        <div className="flex flex-wrap gap-x-2 items-center mb-4 text-sm">
                                            <p className="font-medium text-gray-700">{activeCitation.source}</p>
                                            <span className="text-gray-400">•</span>
                                            <p className="text-gray-600">{activeCitation.date}</p>
                                        </div>
                                        <div className="prose prose-sm max-w-none text-gray-700 mb-6"> <p>{activeCitation.preview}</p> </div>
                                        <a href={activeCitation.url} target="_blank" rel="noopener noreferrer" className="text-sm font-medium flex items-center gap-1 transition-colors duration-150 hover:underline" style={{ color: "#008075" }}> View original source <ExternalLink size={14} className="inline-block"/> </a>
                                    </div>
                                </div>
                            ) : (
                                <div className="h-full flex items-center justify-center text-center text-gray-500">
                                    <div className="p-8">
                                        <div className="rounded-full p-4 bg-[#e6f2f1] inline-flex items-center justify-center mb-4"> <Book size={24} style={{ color: "#008075" }} /> </div>
                                        <h3 className="text-lg font-medium text-gray-700 mb-2">No citation selected</h3>
                                        <p className="text-sm text-gray-500 max-w-xs mx-auto"> Click on a citation number [1] in the chat or select one from the list above to view details. </p>
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                </div>

            </div>

            {/* ================= Knowledge Base Sidebar ================= */}
            <Sidebar
                visible={knowledgeBaseVisible}
                onHide={() => setKnowledgeBaseVisible(false)}
                fullScreen
                className="bg-gray-100" // Set background for the sidebar container
            >
                <div className="flex flex-col h-full">
                    {/* Sidebar Header */}
                    <div className="flex justify-between items-center p-4 bg-white border-b border-gray-300 flex-shrink-0">
                         <button onClick={() => setKnowledgeBaseVisible(false)} className="p-2 rounded-full text-gray-500 hover:bg-gray-200 focus:outline-none focus:ring-2 focus:ring-indigo-500 mr-2" aria-label="Close Knowledge Base">
                             <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6"> <path strokeLinecap="round" strokeLinejoin="round" d="M6 18 18 6M6 6l12 12" /> </svg>
                         </button>
                        <h2 className="text-xl font-semibold text-gray-800">Knowledge Base Documents</h2>
                         <Button label="Upload Document" icon="pi pi-upload" className="p-button-sm p-button-outlined" onClick={handleUploadDocument} />
                    </div>

                    {/* Table Area */}
                    <div className="flex-grow p-4 overflow-auto">
                        <DataTable
                            value={knowledgeBaseDocs}
                            className="p-datatable-sm shadow-md rounded-lg"
                            paginator rows={10} rowsPerPageOptions={[5, 10, 25, 50]}
                            dataKey="id"
                            emptyMessage="No documents found."
                            resizableColumns columnResizeMode="fit"
                            showGridlines
                         >
                            <Column field="srNo" header="Sr No" sortable style={{ width: '5rem', textAlign: 'center' }} />
                            <Column field="fileName" header="File Name" sortable filter filterPlaceholder="Search by name" style={{ minWidth: '20rem' }} />
                            <Column field="lastModified" header="Last Modified" sortable style={{ width: '10rem' }} />
                            <Column field="size" header="Size" sortable style={{ width: '8rem' }}/>
                            <Column header="Actions" body={documentActionBodyTemplate} style={{ width: '8rem', textAlign: 'center' }} />
                        </DataTable>
                    </div>
                </div>
            </Sidebar>

        </div> // End of main application div
    );
};

export default IntegratedChatApp;
