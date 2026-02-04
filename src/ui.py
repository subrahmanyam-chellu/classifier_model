import streamlit as st
import requests
from pymongo import MongoClient
from streamlit_cookies_manager import EncryptedCookieManager
from dotenv import load_dotenv
import os


def validate_issue_text(issue: str) -> str | None:
    # Minimum length check
    if len(issue.strip()) < 15:
        return "Issue must be at least 15 characters long."
    
    # Simple relevance check: reject generic greetings
    invalid_phrases = ["hello", "hi", "how are you", "good morning", "good evening"]
    for phrase in invalid_phrases:
        if phrase in issue.lower():
            return "Please describe a real issue, not just a greeting."
    
    return None  # valid

# --- Cookie Setup ---
cookies = EncryptedCookieManager(
    prefix="ticket_app",
    password="a-very-secret-password"  # change this to something secure
)

if not cookies.ready():
    st.stop()

# --- MongoDB Setup ---

load_dotenv()
uri = os.getenv("Mongo_Atlas")

client = MongoClient(uri)
db = client["ticket_system"]
users_collection = db["users"]
tickets_collection = db["tickets"]

# --- Restore login state from cookies ---
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = cookies.get("logged_in") == "True"
if "role" not in st.session_state:
    st.session_state["role"] = cookies.get("role")
if "current_user" not in st.session_state:
    st.session_state["current_user"] = cookies.get("current_user")

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")

if not st.session_state["logged_in"]:
    page = st.sidebar.radio("Go to:", ["Login", "Register"])
else:
    if st.session_state["role"] == "user":
        page = st.sidebar.radio("Go to:", ["Home", "My Tickets", "Logout"])
    elif st.session_state["role"] == "agent":
        page = st.sidebar.radio("Go to:", ["All Tickets", "Logout"])

# --- Registration Page ---
if page == "Register":
    st.title("Register 📝")
    with st.form("register_form"):
        name = st.text_input("Name")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        role = st.selectbox("Role", ["user", "agent"])
        submitted = st.form_submit_button("Register")

        if submitted:
            if users_collection.find_one({"email": email}):
                st.error("Email already registered.")
            else:
                users_collection.insert_one({
                    "name": name,
                    "email": email,
                    "password": password,
                    "role": role
                })
                st.success("Registration successful! Please login.")

# --- Login Page ---
elif page == "Login":
    st.title("Login 🔑")
    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            user = users_collection.find_one({"email": email, "password": password})
            if user:
                st.session_state["logged_in"] = True
                st.session_state["role"] = user["role"]
                st.session_state["current_user"] = email

                # Save to cookies
                cookies["logged_in"] = "True"
                cookies["role"] = user["role"]
                cookies["current_user"] = email
                cookies.save()

                st.success(f"Welcome {user['name']}! You are logged in as {user['role']}.")
                st.rerun()
            else:
                st.error("Invalid credentials.")

# --- Logout ---
elif page == "Logout":
    st.session_state["logged_in"] = False
    st.session_state["role"] = None
    st.session_state["current_user"] = None

    # Clear cookies
    cookies["logged_in"] = "False"
    cookies["role"] = ""
    cookies["current_user"] = ""
    cookies.save()

    st.info("You have been logged out.")

# --- User Home Page ---
elif page == "Home" and st.session_state["role"] == "user":
    st.title("Welcome to the ML Ticket System 🎫")
    

    with st.form("user_form"):
        title = st.text_input("Enter the title")
        issue = st.text_area("Describe your issue")
        submitted = st.form_submit_button("Submit")

        if submitted:
            error_msg = validate_issue_text(issue)
            if error_msg:
                st.error(error_msg)  # show validation error
            response = requests.post("http://127.0.0.1:8000/predict", json={"issue": issue})
            if response.status_code == 200:
                ticketGet = response.json()
                user = users_collection.find_one({"email": st.session_state["current_user"]})
                tickets_collection.insert_one({
                    "name": user["name"],
                    "email": st.session_state["current_user"],
                    "title": title,
                    "ticket_id": ticketGet["ticket_id"],
                    "issue": issue,
                    "category": ticketGet["predicted_label"][0],
                    "priority": ticketGet["predicted_label"][1],
                    "time_stamp": ticketGet["tStamp"],
                    "status": "open"
                })
                st.success("Your ticket has been generated.")
            else:
                st.error(f"Failed to fetch ticket from FastAPI backend. {response.text}")

# --- User Tickets Page ---
elif page == "My Tickets" and st.session_state["role"] == "user":
    st.title("My Tickets 📋")
    user_tickets = list(tickets_collection.find({"email": st.session_state["current_user"]}, {"_id": 0}))
    if user_tickets:
        # Table header
        cols = st.columns([1, 2, 2, 2, 2, 2, 2, 2])
        headers = ["ID", "Title", "Category", "Priority", "Status", "Issue", "time", "Created By"]
        for col, header in zip(cols, headers):
            col.markdown(f"**{header}**")

        # Table rows
        for ticket in user_tickets:
            cols = st.columns([1, 2, 2, 2, 2, 2, 2, 2])
            cols[0].write(ticket["ticket_id"])
            cols[1].write(ticket["title"])
            cols[2].write(ticket["category"])
            cols[3].write(ticket["priority"])
            cols[4].write(ticket.get("status", "open"))
            cols[5].write(ticket["issue"])
            cols[6].write(ticket.get("time_stamp", "N/A"))
            cols[7].write(ticket["name"])
    else:
        st.info("No tickets generated yet.")


# --- Agent Tickets Page ---
# --- Agent Tickets Page ---
elif page == "All Tickets" and st.session_state["role"] == "agent":
    st.title("All Tickets (Agent View) 📋")

    all_tickets = list(tickets_collection.find({}, {"_id": 0}))
    if all_tickets:
        # Added Issue column and styled Delete button
        cols = st.columns([2, 2, 4, 4, 4, 4, 3, 4, 4])
        headers = ["ID", "Email", "Title", "Issue", "Category", "Priority", "Time_stamp", "Status",  "Action"]
        for col, header in zip(cols, headers):
            col.markdown(f"**{header}**")

        for ticket in all_tickets:
            cols = st.columns([2, 2, 4, 4, 4, 4, 3, 4,4])
            cols[0].write(ticket["ticket_id"])
            #cols[1].write(ticket["name"])
            cols[1].write(ticket["email"])
            cols[2].write(ticket["title"])
            cols[3].write(ticket["issue"])
            cols[4].write(ticket["category"])
            cols[5].write(ticket["priority"])
            cols[6].write(ticket.get("time_stamp", "N/A"))

            current_status = ticket.get("status", "open")
            new_status = cols[7].selectbox(
                "",
                ["open", "in-progress", "resolved", "closed"],
                index=["open", "in-progress", "resolved", "closed"].index(current_status),
                key=f"status_{ticket['ticket_id']}"
            )

            if new_status != current_status:
                tickets_collection.update_one(
                    {"ticket_id": ticket["ticket_id"]},
                    {"$set": {"status": new_status}}
                )
                st.success(f"Ticket {ticket['ticket_id']} status updated to {new_status}")
                st.rerun()

            # Show issue text
            #cols[6].write(ticket["issue"])

            # ✅ Better Delete button
            if cols[8].button("Delete", key=f"delete_{ticket['ticket_id']}"):
                tickets_collection.delete_one({"ticket_id": ticket["ticket_id"]})
                st.warning(f"Ticket {ticket['ticket_id']} deleted.")
                st.rerun()
    else:
        st.info("No tickets available.")