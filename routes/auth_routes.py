from flask import Blueprint, request, render_template, session, redirect, flash, get_flashed_messages
from services.auth_service import register_user, login_user, get_user_by_email

auth = Blueprint('auth', __name__)

@auth.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        # Get flashed messages if any
        return render_template("register.html", messages=get_flashed_messages())  

    data = request.form
    print("FORM DATA:", data)

    name = data.get('name')
    email = data.get('email')
    password = data.get('password')

    if not all([name, email, password]):
        flash("Missing fields! Please fill all required inputs.")
        return redirect('/register')
    
    existing_user = get_user_by_email(email)  

    if existing_user:
        flash("User already exists! Please login or use another email.")
        return redirect('/register')  

    register_user(name, email, password)
    flash("Registration successful! You can now login.")
    return redirect('/login')

@auth.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template("login.html", messages=get_flashed_messages())

    data = request.form
    email = data.get('email')
    password = data.get('password')

    if not all([email, password]):
        flash("Missing fields! Please enter both email and password.")
        return redirect('/login')

    user = get_user_by_email(email)
    if not user:
        flash("User not found! Please register first.")
        return redirect('/login')

    if not login_user(email, password): 
        flash("Incorrect password! Try again.")
        return redirect('/login')

    session['user_email'] = email
    flash(f"Welcome back, {email}!")
    return redirect('/user')
    
@auth.route('/logout')
def logout():
    session.clear()
    flash("You have been logged out successfully.")
    return redirect('/login')

