"""Real-time collaboration hub for live coding sessions."""

import asyncio
import json
import logging
import uuid
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
import websockets
from websockets.server import WebSocketServerProtocol

logger = logging.getLogger(__name__)

@dataclass
class CollaborationSession:
    """Real-time collaboration session."""
    session_id: str
    participants: Set[str] = field(default_factory=set)
    code_state: str = ""
    messages: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    active: bool = True

@dataclass
class CollaborationMessage:
    """Message in collaboration session."""
    message_id: str
    session_id: str
    sender: str
    message_type: str  # "code_change", "cursor", "comment", "join", "leave"
    content: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)

class CollaborationHub:
    """Manages real-time collaboration sessions."""
    
    def __init__(self):
        self.sessions: Dict[str, CollaborationSession] = {}
        self.connections: Dict[str, WebSocketServerProtocol] = {}
        self.session_participants: Dict[str, Set[str]] = {}
    
    def create_session(self, creator: str) -> str:
        """Create new collaboration session."""
        session_id = str(uuid.uuid4())
        session = CollaborationSession(session_id=session_id)
        session.participants.add(creator)
        self.sessions[session_id] = session
        self.session_participants[session_id] = {creator}
        logger.info(f"Created session {session_id} by {creator}")
        return session_id
    
    def join_session(self, session_id: str, user: str) -> bool:
        """Join existing session."""
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        if not session.active:
            return False
        
        session.participants.add(user)
        self.session_participants[session_id].add(user)
        self._broadcast_message(session_id, {
            "type": "user_joined",
            "user": user,
            "participants": list(session.participants)
        })
        return True
    
    def leave_session(self, session_id: str, user: str):
        """Leave session."""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.participants.discard(user)
            self.session_participants[session_id].discard(user)
            
            self._broadcast_message(session_id, {
                "type": "user_left",
                "user": user,
                "participants": list(session.participants)
            })
            
            # End session if no participants
            if not session.participants:
                session.active = False
    
    def update_code(self, session_id: str, user: str, code: str):
        """Update code in session."""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.code_state = code
            
            self._broadcast_message(session_id, {
                "type": "code_update",
                "user": user,
                "code": code
            }, exclude_user=user)
    
    def send_message(self, session_id: str, user: str, message: str):
        """Send chat message."""
        if session_id in self.sessions:
            self._broadcast_message(session_id, {
                "type": "chat_message",
                "user": user,
                "message": message
            })
    
    def _broadcast_message(self, session_id: str, message: Dict[str, Any], exclude_user: Optional[str] = None):
        """Broadcast message to session participants."""
        if session_id not in self.session_participants:
            return
        
        message["timestamp"] = datetime.now().isoformat()
        
        for user in self.session_participants[session_id]:
            if user == exclude_user:
                continue
            if user in self.connections:
                try:
                    asyncio.create_task(self._send_to_user(user, message))
                except Exception as e:
                    logger.error(f"Failed to send to {user}: {e}")
    
    async def _send_to_user(self, user: str, message: Dict[str, Any]):
        """Send message to specific user."""
        if user in self.connections:
            conn = self.connections[user]
            await conn.send(json.dumps(message))
    
    def register_connection(self, user: str, connection: WebSocketServerProtocol):
        """Register WebSocket connection."""
        self.connections[user] = connection
    
    def unregister_connection(self, user: str):
        """Unregister connection."""
        self.connections.pop(user, None)
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session information."""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        return {
            "session_id": session_id,
            "participants": list(session.participants),
            "active": session.active,
            "created_at": session.created_at.isoformat(),
            "code_length": len(session.code_state)
        }

# WebSocket handler
async def collaboration_handler(websocket: WebSocketServerProtocol, path: str):
    """Handle WebSocket connections for collaboration."""
    hub = CollaborationHub()  # In practice, use shared instance
    
    user = None
    current_session = None
    
    try:
        async for message in websocket:
            data = json.loads(message)
            action = data.get("action")
            
            if action == "join":
                user = data["user"]
                session_id = data["session_id"]
                hub.register_connection(user, websocket)
                
                if hub.join_session(session_id, user):
                    current_session = session_id
                    await websocket.send(json.dumps({
                        "type": "joined",
                        "session": hub.get_session_info(session_id)
                    }))
                else:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": "Failed to join session"
                    }))
            
            elif action == "create_session":
                user = data["user"]
                hub.register_connection(user, websocket)
                session_id = hub.create_session(user)
                current_session = session_id
                await websocket.send(json.dumps({
                    "type": "session_created",
                    "session_id": session_id
                }))
            
            elif action == "update_code" and current_session and user:
                hub.update_code(current_session, user, data["code"])
            
            elif action == "send_message" and current_session and user:
                hub.send_message(current_session, user, data["message"])
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if user:
            hub.unregister_connection(user)
            if current_session:
                hub.leave_session(current_session, user)

# Example usage
def start_collaboration_server():
    """Start WebSocket server for collaboration."""
    start_server = websockets.serve(collaboration_handler, "localhost", 8765)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()

if __name__ == "__main__":
    start_collaboration_server()
