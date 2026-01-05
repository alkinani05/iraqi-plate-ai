import streamlit as st
import hashlib
import os

# ---------------------------------------------------------------------
# 🔐 AUTHENTICATION MODULE
# ---------------------------------------------------------------------

def get_credentials():
    """
    Fetch credentials from Environment variables or default to secure values.
    NEVER hardcode production passwords.
    """
    admin_user = os.getenv("APP_USER", "husam")
    # Default hash for "987987987" - Change this in production!
    default_hash = "5d5069b6272504287535c5c088c4b7754d9c73e164478332152bd6204005088c"
    admin_pass_hash = os.getenv("APP_PASS_HASH", default_hash)
    return admin_user, admin_pass_hash

def check_password():
    """
    Secure password check with session state management.
    Returns True if authenticated.
    """
    ADMIN_USER, ADMIN_PASS_HASH = get_credentials()

    def password_entered():
        if (st.session_state["username"].strip() == ADMIN_USER and 
            hashlib.sha256(st.session_state["password"].encode()).hexdigest() == ADMIN_PASS_HASH):
            st.session_state["authenticated"] = True
            del st.session_state["password"]  # Don't store password in RAM
        else:
            st.session_state["authenticated"] = False

    if "authenticated" not in st.session_state:
        # First Run
        st.text_input("Username / اسم المستخدم", key="username")
        st.text_input("Password / كلمة المرور", type="password", key="password", on_change=password_entered)
        return False
    elif not st.session_state["authenticated"]:
        # Failed Attempt
        st.text_input("Username / اسم المستخدم", key="username")
        st.text_input("Password / كلمة المرور", type="password", key="password", on_change=password_entered)
        st.error("🔒 Incorrect credentials / بيانات خاطئة")
        return False
    else:
        # Success
        return True
