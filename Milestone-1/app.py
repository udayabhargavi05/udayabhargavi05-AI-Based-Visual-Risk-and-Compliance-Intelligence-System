# app.py
from flask import Flask, render_template, request, redirect, url_for, flash, session
from jinja2 import ChoiceLoader, DictLoader
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    login_required,
    logout_user,
    current_user,
)
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import numpy as np
from sklearn.linear_model import LinearRegression
import os
from dotenv import load_dotenv
from collections import defaultdict

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-key-change-in-production')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///digital_twin.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'


# ------------------------- Context Processor -------------------------
@app.context_processor
def utility_processor():
    return dict(datetime=datetime)


# ------------------------- Gemini AI Setup -------------------------
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

if not GEMINI_AVAILABLE:
    print(
        'Warning: google-generativeai not installed. '
        'Install with: pip install google-generativeai'
    )

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

if GEMINI_AVAILABLE and GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
    print('Gemini initialized successfully.')
else:
    gemini_model = None
    if not GEMINI_API_KEY:
        print(
            'Warning: GEMINI_API_KEY not set. '
            'Assistant will use rule-based responses.'
        )


# ------------------------- Database Models -------------------------
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    age = db.Column(db.Integer, default=25)
    occupation = db.Column(db.String(100), default='Student')
    monthly_income = db.Column(db.Float, default=50000)
    monthly_expenses = db.Column(db.Float, default=35000)
    savings = db.Column(db.Float, default=100000)
    study_hours_per_week = db.Column(db.Float, default=20)
    fitness_hours_per_week = db.Column(db.Float, default=5)
    sleep_hours_per_day = db.Column(db.Float, default=7)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_admin = db.Column(db.Boolean, default=False)

    transactions = db.relationship('Transaction', backref='user', lazy=True)
    study_logs = db.relationship('StudyLog', backref='user', lazy=True)
    fitness_logs = db.relationship('FitnessLog', backref='user', lazy=True)
    goals = db.relationship('Goal', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Transaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    date = db.Column(db.Date, nullable=False, default=datetime.utcnow)
    category = db.Column(db.String(50))
    amount = db.Column(db.Float, nullable=False)
    type = db.Column(db.String(10))


class StudyLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    date = db.Column(db.Date, nullable=False, default=datetime.utcnow)
    hours = db.Column(db.Float, nullable=False)
    subject = db.Column(db.String(100))
    productivity_score = db.Column(db.Integer)


class FitnessLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    date = db.Column(db.Date, nullable=False, default=datetime.utcnow)
    activity = db.Column(db.String(50))
    duration_min = db.Column(db.Integer, nullable=False)
    calories_burned = db.Column(db.Integer)


class Goal(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    description = db.Column(db.String(200))
    target_date = db.Column(db.Date)
    achieved = db.Column(db.Boolean, default=False)


class ChatMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    question = db.Column(db.Text, nullable=False)
    response = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user = db.relationship('User', backref='chat_messages')


class SimulationHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    scenario = db.Column(db.String(50), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    result_desc = db.Column(db.String(200))
    result_impact = db.Column(db.String(200))
    result_risk = db.Column(db.String(20))
    result_rec = db.Column(db.String(200))
    projected = db.Column(db.Float)
    current = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user = db.relationship('User', backref='simulations')


# ------------------------- Helper Functions -------------------------
def generate_forecast(data, periods=6):
    if len(data) < 2:
        return [data[-1] if data else 0] * periods

    X = np.array(range(len(data))).reshape(-1, 1)
    y = np.array(data)
    model = LinearRegression()
    model.fit(X, y)

    future_X = np.array(
        range(len(data), len(data) + periods)
    ).reshape(-1, 1)
    forecast = model.predict(future_X)
    noise = np.random.normal(
        0,
        np.std(y) * 0.1 if np.std(y) > 0 else 1,
        periods,
    )
    return (forecast + noise).tolist()


def calculate_financial_metrics(user):
    income = sum(t.amount for t in user.transactions if t.type == 'income')
    expenses = sum(t.amount for t in user.transactions if t.type == 'expense')
    net = income - expenses
    rate = (net / income * 100) if income > 0 else 0
    return {
        'income': income,
        'expenses': expenses,
        'net': net,
        'rate': rate,
    }


def predict_gpa(user):
    logs = user.study_logs
    if not logs:
        return {
            'gpa': 3.0,
            'efficiency': 70,
            'recommendation': 'Log study sessions to get predictions.',
        }

    total_hours = sum(l.hours for l in logs)
    avg_prod = sum((l.productivity_score or 0) for l in logs) / len(logs)
    gpa = 2.0 + (total_hours / 100) + (avg_prod / 100)
    gpa = min(4.0, max(1.0, gpa))

    if total_hours < 20:
        rec = 'Increase study hours.'
    elif avg_prod < 70:
        rec = 'Improve focus and quality.'
    else:
        rec = 'Keep up the good work!'

    return {
        'gpa': round(gpa, 2),
        'efficiency': round(avg_prod, 1),
        'recommendation': rec,
    }


def simulate_scenario(user, action, amount):
    scenarios = {
        'save': {
            'desc': f'Save ₹{amount:,} monthly',
            'impact': f'Your savings grow to ₹{user.savings + amount * 12:,.0f} in 1 year.',
            'risk': 'Low',
            'rec': 'Great for long-term security',
        },
        'invest': {
            'desc': f'Invest ₹{amount:,} monthly',
            'impact': f'With 10% returns, investment grows to ₹{amount * 12 * 1.1:,.0f} in 1 year.',
            'risk': 'Medium',
            'rec': 'Balanced approach for growth',
        },
        'spend': {
            'desc': f'Spend ₹{amount:,} monthly extra',
            'impact': f'Your savings reduce to ₹{user.savings - amount * 12:,.0f} in 1 year.',
            'risk': 'High',
            'rec': 'Consider reducing discretionary spending',
        },
        'study_more': {
            'desc': f'Study {amount} more hours per week',
            'impact': f'Your GPA could increase by {amount * 0.1:.2f} points.',
            'risk': 'Low',
            'rec': 'Consistent effort yields results',
        },
        'exercise': {
            'desc': f'Exercise {amount} minutes daily',
            'impact': f'You could burn {amount * 30 * 30:,.0f} calories monthly.',
            'risk': 'Low',
            'rec': 'Great for overall health',
        },
    }
    return scenarios.get(action, scenarios['save'])


# ------------------------- Default Admin Creator -------------------------
def create_default_admin():
    """Create a default admin user if none exists."""
    admin = User.query.filter_by(username='admin').first()
    if not admin:
        admin = User(
            username='admin',
            email='admin@digitaltwin.ai',
        )
        admin.set_password('admin123')
        admin.is_admin = True
        db.session.add(admin)
        db.session.commit()
        print('✅ Default admin user created:')
        print(' Username: admin')
        print(' Password: admin123')
        print(' Please change your password after first login.')
    else:
        print('✅ Admin user already exists.')


# ------------------------- Inline templates (keeps project to 3 files) -------------------------
# These templates are stored in app.py so login, register and feature pages work
# without requiring additional HTML files in the templates directory.
INTERNAL_TEMPLATES = {
    'user/dashboard.html': r"""{% extends 'base.html' %}{% block content %}<div class="container py-4"><h1>Welcome, {{ user.username|e }}</h1><p class="text-muted">Your personal dashboard</p><div class="row g-3 my-2"><div class="col-md-3"><div class="card p-3"><small>Income logged</small><h3>₹{{ '%.2f'|format(metrics.income) }}</h3></div></div><div class="col-md-3"><div class="card p-3"><small>Expenses logged</small><h3>₹{{ '%.2f'|format(metrics.expenses) }}</h3></div></div><div class="col-md-3"><div class="card p-3"><small>Study hours</small><h3>{{ '%.1f'|format(total_study_hours) }}</h3></div></div><div class="col-md-3"><div class="card p-3"><small>Fitness minutes</small><h3>{{ total_fitness }}</h3></div></div></div><div class="card p-3 my-3"><h4>Life snapshot</h4><p>Estimated GPA: <b>{{ gpa.gpa }}</b> · Life score: <b>{{ life_score }}</b> · Study streak: <b>{{ study_streak }} days</b> · Fitness streak: <b>{{ fitness_streak }} days</b></p><p>{{ gpa.recommendation }}</p></div><div class="d-flex flex-wrap gap-2"><a class="btn btn-primary" href="{{ url_for('user_financial') }}">Financial</a><a class="btn btn-primary" href="{{ url_for('user_study') }}">Study</a><a class="btn btn-primary" href="{{ url_for('user_fitness') }}">Fitness</a><a class="btn btn-primary" href="{{ url_for('user_simulation') }}">Simulation</a><a class="btn btn-primary" href="{{ url_for('user_assistant') }}">AI Assistant</a></div></div>{% endblock %}""",
    'user/financial.html': r"""{% extends 'base.html' %}{% block content %}<div class="container py-4"><h1>Financial Management</h1><div class="row g-3 my-2"><div class="col-md-4"><div class="card p-3">Income <h4>₹{{ '%.2f'|format(metrics.income) }}</h4></div></div><div class="col-md-4"><div class="card p-3">Expenses <h4>₹{{ '%.2f'|format(metrics.expenses) }}</h4></div></div><div class="col-md-4"><div class="card p-3">Net <h4>₹{{ '%.2f'|format(metrics.net) }}</h4></div></div></div><div class="card p-3 my-3"><h4>Add transaction</h4><form method="post" class="row g-2"><div class="col-md-3"><input class="form-control" type="date" name="date" value="{{ datetime.now().strftime('%Y-%m-%d') }}" required></div><div class="col-md-3"><input class="form-control" name="category" placeholder="Category" required maxlength="50"></div><div class="col-md-3"><input class="form-control" type="number" name="amount" min="0.01" step="0.01" placeholder="Amount" required></div><div class="col-md-2"><select class="form-select" name="type"><option value="expense">Expense</option><option value="income">Income</option></select></div><div class="col-md-1"><button class="btn btn-primary">Add</button></div></form></div><div class="card p-3"><h4>Recent transactions</h4><div class="table-responsive"><table class="table"><thead><tr><th>Date</th><th>Category</th><th>Type</th><th>Amount</th><th></th></tr></thead><tbody>{% for tx in transactions %}<tr><td>{{ tx.date }}</td><td>{{ tx.category|e }}</td><td>{{ tx.type|e }}</td><td>₹{{ '%.2f'|format(tx.amount) }}</td><td><form method="post" action="{{ url_for('delete_transaction', id=tx.id) }}" onsubmit="return confirm('Delete this transaction?')"><button class="btn btn-sm btn-outline-danger">Delete</button></form></td></tr>{% else %}<tr><td colspan="5">No transactions yet.</td></tr>{% endfor %}</tbody></table></div></div></div>{% endblock %}""",
    'user/study.html': r"""{% extends 'base.html' %}{% block content %}<div class="container py-4"><h1>Study Intelligence</h1><p>Estimated GPA: <b>{{ gpa.gpa }}</b> · Average productivity: <b>{{ '%.1f'|format(avg_productivity) }}%</b> · Study streak: <b>{{ study_streak }} days</b></p><p>{{ gpa.recommendation }}</p><div class="card p-3 my-3"><h4>Log a study session</h4><form method="post" class="row g-2"><div class="col-md-3"><input class="form-control" type="date" name="date" value="{{ datetime.now().strftime('%Y-%m-%d') }}" required></div><div class="col-md-3"><input class="form-control" name="subject" placeholder="Subject" required maxlength="100"></div><div class="col-md-3"><input class="form-control" type="number" name="hours" min="0.1" max="24" step="0.1" placeholder="Hours" required></div><div class="col-md-2"><input class="form-control" type="number" name="productivity" min="0" max="100" placeholder="Productivity %" required></div><div class="col-md-1"><button class="btn btn-primary">Add</button></div></form></div><div class="card p-3"><h4>Recent study sessions</h4><div class="table-responsive"><table class="table"><thead><tr><th>Date</th><th>Subject</th><th>Hours</th><th>Productivity</th><th></th></tr></thead><tbody>{% for log in logs %}<tr><td>{{ log.date }}</td><td>{{ log.subject|e }}</td><td>{{ log.hours }}</td><td>{{ log.productivity_score }}%</td><td><form method="post" action="{{ url_for('delete_study_log', id=log.id) }}" onsubmit="return confirm('Delete this session?')"><button class="btn btn-sm btn-outline-danger">Delete</button></form></td></tr>{% else %}<tr><td colspan="5">No study sessions yet.</td></tr>{% endfor %}</tbody></table></div></div></div>{% endblock %}""",
    'user/fitness.html': r"""{% extends 'base.html' %}{% block content %}<div class="container py-4"><h1>Fitness Tracking</h1><div class="row g-3 my-2"><div class="col-md-4"><div class="card p-3">Total minutes <h4>{{ total_min }}</h4></div></div><div class="col-md-4"><div class="card p-3">Calories burned <h4>{{ total_cal }}</h4></div></div><div class="col-md-4"><div class="card p-3">Weekly minutes <h4>{{ weekly_min }} / 150</h4></div></div></div><div class="card p-3 my-3"><h4>Log fitness activity</h4><form method="post" class="row g-2"><div class="col-md-3"><input class="form-control" type="date" name="date" value="{{ datetime.now().strftime('%Y-%m-%d') }}" required></div><div class="col-md-3"><input class="form-control" name="activity" placeholder="Activity" required maxlength="50"></div><div class="col-md-2"><input class="form-control" type="number" name="duration" min="1" placeholder="Minutes" required></div><div class="col-md-2"><input class="form-control" type="number" name="calories" min="0" placeholder="Calories" required></div><div class="col-md-2"><button class="btn btn-primary">Add activity</button></div></form></div><div class="card p-3"><h4>Recent activities</h4><div class="table-responsive"><table class="table"><thead><tr><th>Date</th><th>Activity</th><th>Minutes</th><th>Calories</th><th></th></tr></thead><tbody>{% for log in logs %}<tr><td>{{ log.date }}</td><td>{{ log.activity|e }}</td><td>{{ log.duration_min }}</td><td>{{ log.calories_burned or 0 }}</td><td><form method="post" action="{{ url_for('delete_fitness', log_id=log.id) }}" onsubmit="return confirm('Delete this activity?')"><button class="btn btn-sm btn-outline-danger">Delete</button></form></td></tr>{% else %}<tr><td colspan="5">No fitness activities yet.</td></tr>{% endfor %}</tbody></table></div></div></div>{% endblock %}""",
    'info.html': r"""{% extends 'base.html' %}{% block content %}<div class="container py-5"><div class="card p-4"><h1>{{ page_title|e }}</h1><p>{{ page_text|e }}</p></div></div>{% endblock %}""",
    'admin/dashboard.html': r"""{% extends 'base.html' %}{% block content %}<div class="container py-4"><h1>Admin dashboard</h1><div class="card p-4"><p>Total registered users: <b>{{ user_count }}</b></p><a class="btn btn-primary" href="{{ url_for('admin_users') }}">Manage users</a></div></div>{% endblock %}""",
    'admin/users.html': r"""{% extends 'base.html' %}{% block content %}<div class="container py-4"><h1>Manage users</h1><div class="card p-3 table-responsive"><table class="table"><thead><tr><th>ID</th><th>Username</th><th>Email</th><th>Admin</th><th>Joined</th></tr></thead><tbody>{% for u in users %}<tr><td>{{ u.id }}</td><td>{{ u.username|e }}</td><td>{{ u.email|e }}</td><td>{{ 'Yes' if u.is_admin else 'No' }}</td><td>{{ u.created_at.strftime('%Y-%m-%d') if u.created_at else '' }}</td></tr>{% endfor %}</tbody></table></div></div>{% endblock %}""",
    'user/simulation.html': r"""{% extends 'base.html' %}{% block content %}<div class="container py-4"><h1>Future Simulation</h1><p>Explore a simple estimate of how a choice may affect your future. Estimates are illustrative, not financial advice.</p><div class="card p-3 my-3"><form method="post" class="row g-3"><div class="col-md-5"><label class="form-label">Scenario</label><select class="form-select" name="scenario"><option value="save">Save monthly</option><option value="invest">Invest monthly</option><option value="spend">Spend extra monthly</option><option value="study_more">Study more hours weekly</option><option value="exercise">Exercise minutes daily</option></select></div><div class="col-md-5"><label class="form-label">Amount (₹ for money scenarios; units for time scenarios)</label><input class="form-control" type="number" name="amount" min="1" step="any" required></div><div class="col-md-2 d-flex align-items-end"><button class="btn btn-primary">Simulate</button></div></form></div>{% if result %}<div class="card p-4"><h3>{{ result.desc|e }}</h3><p>{{ result.impact|e }}</p><p>Risk: <b>{{ result.risk|e }}</b></p><p>{{ result.rec|e }}</p></div>{% endif %}<div class="card p-3 mt-3"><h4>Previous simulations</h4><ul>{% for item in history %}<li>{{ item.timestamp.strftime('%Y-%m-%d %H:%M') }} — {{ item.result_desc|e }}: {{ item.result_impact|e }}</li>{% else %}<li>No simulations yet.</li>{% endfor %}</ul></div></div>{% endblock %}""",
    'user/assistant.html': r"""{% extends 'base.html' %}{% block content %}<div class="container py-4" style="max-width:900px"><h1>AI Assistant</h1><p>Ask for general suggestions about finance, studying, fitness, or habits. Responses are informational.</p><div class="card p-3 mb-3"><form method="post"><label class="form-label">Your question</label><textarea class="form-control mb-3" name="question" rows="3" required maxlength="2000">{{ question or '' }}</textarea><button class="btn btn-primary">Ask assistant</button></form></div>{% if answer %}<div class="card p-3 mb-3"><h5>Response</h5><p style="white-space:pre-wrap">{{ answer|e }}</p></div>{% endif %}<div class="card p-3"><h5>Recent conversation</h5>{% for msg in messages %}<div class="border-bottom py-2"><b>You:</b> {{ msg.question|e }}<br><b>Assistant:</b> {{ msg.response|e }}</div>{% else %}<p>No messages yet.</p>{% endfor %}</div></div>{% endblock %}""",
    'settings.html': r"""{% extends 'base.html' %}{% block content %}<div class="container py-4" style="max-width:700px"><h1>Account settings</h1><div class="card p-4"><form method="post"><label class="form-label">Username</label><input class="form-control mb-3" name="username" value="{{ current_user.username|e }}" required maxlength="80"><label class="form-label">Email</label><input class="form-control mb-3" type="email" name="email" value="{{ current_user.email|e }}" required maxlength="120"><label class="form-label">Age</label><input class="form-control mb-3" type="number" name="age" min="13" max="120" value="{{ current_user.age }}"><label class="form-label">Occupation</label><input class="form-control mb-3" name="occupation" value="{{ current_user.occupation|e }}" maxlength="100"><button class="btn btn-primary">Save settings</button></form></div></div>{% endblock %}""",
}
app.jinja_loader = ChoiceLoader([DictLoader(INTERNAL_TEMPLATES), app.jinja_loader])

# ------------------------- Routes -------------------------
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')

        if not username or not email or not password:
            flash('All fields are required.')
            return redirect(url_for('register'))

        if password != confirm_password:
            flash('Passwords do not match.')
            return redirect(url_for('register'))

        if len(password) < 8:
            flash('Password must be at least 8 characters.')
            return redirect(url_for('register'))

        if User.query.filter_by(username=username).first():
            flash('Username already exists.')
            return redirect(url_for('register'))

        if User.query.filter_by(email=email).first():
            flash('Email already registered.')
            return redirect(url_for('register'))

        user = User(username=username, email=email)
        user.set_password(password)
        # First user becomes admin – but now we already have a default admin, so this is optional
        # if User.query.count() == 0:
        #     user.is_admin = True
        db.session.add(user)
        db.session.commit()
        flash('Registration successful! Please log in.')
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        if not username or not password:
            flash('Please enter both username and password.')
            return render_template('login.html')

        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            return redirect(
                url_for('admin_dashboard' if user.is_admin else 'user_dashboard')
            )

        flash('Invalid username or password.')
        return render_template('login.html')

    # GET request: show the login form.
    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))


# ------------------------- User Routes (unchanged) -------------------------
@app.route('/user/dashboard')
@login_required
def user_dashboard():
    # ===== Existing metrics =====
    metrics = calculate_financial_metrics(current_user)
    gpa_info = predict_gpa(current_user)
    total_fitness = sum(f.duration_min for f in current_user.fitness_logs)

    # ===== Chart Data Preparation (as before) =====
    monthly_income = defaultdict(float)
    monthly_expenses = defaultdict(float)
    for tx in current_user.transactions:
        month_key = tx.date.strftime('%Y-%m')
        if tx.type == 'income':
            monthly_income[month_key] += tx.amount
        else:
            monthly_expenses[month_key] += tx.amount

    all_months = sorted(set(monthly_income.keys()) | set(monthly_expenses.keys()))
    months = all_months[-6:]
    income_data = [monthly_income.get(m, 0) for m in months]
    expense_data = [monthly_expenses.get(m, 0) for m in months]

    study_dates = defaultdict(float)
    for log in current_user.study_logs:
        study_dates[log.date.strftime('%Y-%m-%d')] += log.hours
    study_days = sorted(study_dates.keys())[-14:]
    study_hours = [study_dates.get(d, 0) for d in study_days]

    fitness_dates = defaultdict(float)
    for log in current_user.fitness_logs:
        fitness_dates[log.date.strftime('%Y-%m-%d')] += log.duration_min
    fitness_days = sorted(fitness_dates.keys())[-14:]
    fitness_minutes = [fitness_dates.get(d, 0) for d in fitness_days]

    goal_progress = []
    for goal in current_user.goals:
        if not goal.achieved and goal.target_date:
            days_remaining = (goal.target_date - datetime.now().date()).days
            if days_remaining < 0:
                progress = 100
            else:
                progress = min(100, max(0, 100 - (days_remaining / 30 * 100)))
            goal_progress.append({
                'description': goal.description,
                'progress': round(progress, 1),
            })

    # ===== PREDICTIVE ANALYTICS =====
    # Savings forecast (same as before)
    monthly_net = defaultdict(float)
    for tx in current_user.transactions:
        month_key = tx.date.strftime('%Y-%m')
        if tx.type == 'income':
            monthly_net[month_key] += tx.amount
        else:
            monthly_net[month_key] -= tx.amount

    sorted_months = sorted(monthly_net.keys())
    net_values = [monthly_net[m] for m in sorted_months[-6:]]

    if len(net_values) >= 3:
        forecast_net = generate_forecast(net_values, 6)
        savings_forecast = []
        cum = current_user.savings
        for val in forecast_net:
            cum += val
            savings_forecast.append(cum)
    else:
        savings_forecast = [current_user.savings + i * 5000 for i in range(6)]

    # GPA forecast
    study_hours_by_month = defaultdict(float)
    for log in current_user.study_logs:
        month_key = log.date.strftime('%Y-%m')
        study_hours_by_month[month_key] += log.hours

    study_hours_list = [
        study_hours_by_month[m]
        for m in sorted(study_hours_by_month.keys())[-4:]
    ]

    if len(study_hours_list) >= 3:
        forecast_hours = generate_forecast(study_hours_list, 3)
        gpa_forecast = []
        avg_prod = (
            sum((l.productivity_score or 0) for l in current_user.study_logs)
            / len(current_user.study_logs)
            if current_user.study_logs
            else 70
        )
        for hours in forecast_hours:
            gpa = 2.0 + (hours / 50) + (avg_prod / 200)
            gpa_forecast.append(round(min(4.0, max(1.0, gpa)), 2))
    else:
        gpa_forecast = [gpa_info['gpa'] + i * 0.05 for i in range(3)]

    # Fitness projection – use the same logic as before
    fitness_by_day = defaultdict(float)
    for log in current_user.fitness_logs:
        fitness_by_day[log.date.strftime('%Y-%m-%d')] += log.duration_min

    fitness_days_list = list(fitness_by_day.values())[-14:]
    if len(fitness_days_list) >= 5:
        forecast_fitness = generate_forecast(fitness_days_list, 30)
        projected_fitness = sum(forecast_fitness)
    else:
        projected_fitness = total_fitness * 2

    # ===== EXTRA METRICS =====
    total_study_hours = sum(log.hours for log in current_user.study_logs) or 0

    # Study streak (consecutive days with at least some study in last 7 days)
    study_streak = 0
    if current_user.study_logs:
        study_days_set = set(log.date for log in current_user.study_logs)
        today = datetime.now().date()
        for i in range(7):
            if today - timedelta(days=i) in study_days_set:
                study_streak += 1
            else:
                break

    # Fitness streak (consecutive days with >= 10 min activity)
    fitness_streak = 0
    if current_user.fitness_logs:
        today = datetime.now().date()
        for i in range(7):
            day_total = sum(
                l.duration_min
                for l in current_user.fitness_logs
                if l.date == (today - timedelta(days=i))
            )
            if day_total >= 10:
                fitness_streak += 1
            else:
                break

    # Subject breakdown
    subject_totals = defaultdict(float)
    for log in current_user.study_logs:
        if log.subject:
            subject_totals[log.subject] += log.hours
    subjects = list(subject_totals.keys())
    subject_hours = list(subject_totals.values())

    # Activity breakdown
    activity_totals = defaultdict(float)
    for log in current_user.fitness_logs:
        activity_totals[log.activity] += log.duration_min
    activities = list(activity_totals.keys())
    activity_minutes = list(activity_totals.values())

    # Life Score
    life_score = (
        (metrics['rate'] / 30) * 20
        + (gpa_info['gpa'] / 4) * 30
        + (min(total_fitness, 300) / 300) * 25
        + (
            len([g for g in current_user.goals if g.achieved])
            / max(1, len(current_user.goals))
        ) * 25
    )
    life_score = round(life_score)

    # ===== RENDER =====
    return render_template(
        'user/dashboard.html',
        user=current_user,
        metrics=metrics,
        gpa=gpa_info,
        total_fitness=total_fitness,
        months=months,
        income_data=income_data,
        expense_data=expense_data,
        study_days=study_days,
        study_hours=study_hours,
        fitness_days=fitness_days,
        fitness_minutes=fitness_minutes,
        goal_progress=goal_progress,
        savings_forecast=savings_forecast,
        gpa_forecast=gpa_forecast,
        projected_fitness=projected_fitness,
        total_study_hours=total_study_hours,
        study_streak=study_streak,
        fitness_streak=fitness_streak,
        subjects=subjects,
        subject_hours=subject_hours,
        activities=activities,
        activity_minutes=activity_minutes,
        life_score=life_score,
    )


@app.route('/user/financial', methods=['GET', 'POST'])
@login_required
def user_financial():
    if request.method == 'POST':
        date = datetime.strptime(request.form['date'], '%Y-%m-%d')
        category = request.form['category']
        amount = float(request.form['amount'])
        trans_type = request.form['type']
        tx = Transaction(
            user_id=current_user.id,
            date=date,
            category=category,
            amount=amount,
            type=trans_type,
        )
        db.session.add(tx)
        db.session.commit()
        flash('Transaction added.')
        return redirect(url_for('user_financial'))

    transactions = current_user.transactions
    expense_data = [t.amount for t in transactions if t.type == 'expense']
    forecast = generate_forecast(expense_data, 6) if expense_data else []

    # Metrics
    metrics = calculate_financial_metrics(current_user)

    # Category breakdown for pie chart (only expenses)
    category_totals = {}
    for tx in transactions:
        if tx.type == 'expense':
            category_totals[tx.category] = (
                category_totals.get(tx.category, 0) + tx.amount
            )

    categories = list(category_totals.keys())
    category_amounts = list(category_totals.values())

    # For the recent transactions list, we'll limit to last 10
    recent_transactions = transactions[-10:][::-1]  # newest first

    return render_template(
        'user/financial.html',
        user=current_user,
        transactions=recent_transactions,
        forecast=forecast,
        metrics=metrics,
        categories=categories,
        category_amounts=category_amounts,
    )


@app.route('/edit_transaction', methods=['POST'])
@login_required
def edit_transaction():
    """Update an existing transaction (called from the modal)."""
    tx_id = request.form.get('id')
    if not tx_id:
        flash('Transaction ID missing.', 'danger')
        return redirect(url_for('user_financial'))

    # Retrieve the transaction and ensure it belongs to the current user
    tx = Transaction.query.filter_by(
        id=tx_id,
        user_id=current_user.id,
    ).first()
    if not tx:
        flash('Transaction not found or you do not have permission.', 'danger')
        return redirect(url_for('user_financial'))

    # Update fields
    try:
        tx.date = datetime.strptime(
            request.form['date'], '%Y-%m-%d'
        ).date()
        tx.category = request.form['category']
        tx.amount = float(request.form['amount'])
        tx.type = request.form['type']
        db.session.commit()
        flash('Transaction updated successfully!', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error updating transaction: {str(e)}', 'danger')

    return redirect(url_for('user_financial'))


@app.route('/delete_transaction/<int:id>', methods=['POST'])
@login_required
def delete_transaction(id):
    """Delete a transaction by ID (only if it belongs to the current user)."""
    tx = Transaction.query.filter_by(
        id=id,
        user_id=current_user.id,
    ).first()
    if not tx:
        flash('Transaction not found or you do not have permission.', 'danger')
        return redirect(url_for('user_financial'))

    try:
        db.session.delete(tx)
        db.session.commit()
        flash('Transaction deleted.', 'info')
    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting transaction: {str(e)}', 'danger')

    return redirect(url_for('user_financial'))


@app.route('/user/study', methods=['GET', 'POST'])
@login_required
def user_study():
    if request.method == 'POST':
        date = datetime.strptime(request.form['date'], '%Y-%m-%d')
        hours = float(request.form['hours'])
        subject = request.form['subject']
        productivity = int(request.form['productivity'])
        log = StudyLog(
            user_id=current_user.id,
            date=date,
            hours=hours,
            subject=subject,
            productivity_score=productivity,
        )
        db.session.add(log)
        db.session.commit()
        flash('Study session logged.', 'success')
        return redirect(url_for('user_study'))

    logs = current_user.study_logs
    gpa_info = predict_gpa(current_user)

    # --- Data for charts ---
    study_by_day = defaultdict(float)
    for log in logs:
        study_by_day[log.date.strftime('%Y-%m-%d')] += log.hours

    study_dates = sorted(study_by_day.keys())[-14:]
    study_hours = [study_by_day.get(d, 0) for d in study_dates]

    subject_totals = defaultdict(float)
    for log in logs:
        if log.subject:
            subject_totals[log.subject] += log.hours

    subjects = list(subject_totals.keys())
    subject_hours = list(subject_totals.values())

    prod_by_day = defaultdict(list)
    for log in logs:
        prod_by_day[log.date.strftime('%Y-%m-%d')].append(
            log.productivity_score
        )

    prod_dates = sorted(prod_by_day.keys())[-14:]
    prod_avg = [
        sum(prod_by_day[d]) / len(prod_by_day[d]) if d in prod_by_day else 0
        for d in prod_dates
    ]

    total_hours = sum(log.hours for log in logs)
    avg_productivity = (
        sum((log.productivity_score or 0) for log in logs) / len(logs)
        if logs
        else 0
    )

    # Study streak
    study_streak = 0
    if logs:
        study_days_set = sorted(set(log.date for log in logs), reverse=True)
        today = datetime.now().date()
        for i in range(7):
            check_date = today - timedelta(days=i)
            if check_date in study_days_set:
                study_streak += 1
            else:
                break

    recent_logs = logs[-10:][::-1]

    return render_template(
        'user/study.html',
        user=current_user,
        logs=recent_logs,
        gpa=gpa_info,
        study_dates=study_dates,
        study_hours=study_hours,
        subjects=subjects,
        subject_hours=subject_hours,
        prod_dates=prod_dates,
        prod_avg=prod_avg,
        total_hours=total_hours,
        avg_productivity=avg_productivity,
        study_streak=study_streak,
    )


@app.route('/edit_study_log', methods=['POST'])
@login_required
def edit_study_log():
    log_id = request.form.get('id')
    if not log_id:
        flash('Log ID missing.', 'danger')
        return redirect(url_for('user_study'))

    log = StudyLog.query.filter_by(
        id=log_id,
        user_id=current_user.id,
    ).first()
    if not log:
        flash('Log not found or you do not have permission.', 'danger')
        return redirect(url_for('user_study'))

    try:
        log.date = datetime.strptime(
            request.form['date'], '%Y-%m-%d'
        ).date()
        log.hours = float(request.form['hours'])
        log.subject = request.form.get('subject')
        log.productivity_score = int(request.form['productivity'])
        db.session.commit()
        flash('Study log updated successfully!', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error updating log: {str(e)}', 'danger')

    return redirect(url_for('user_study'))



@app.route('/delete_study_log/<int:id>', methods=['POST'])
@login_required
def delete_study_log(id):
    log = StudyLog.query.filter_by(
        id=id,
        user_id=current_user.id,
    ).first()
    if not log:
        flash('Log not found or you do not have permission.', 'danger')
        return redirect(url_for('user_study'))

    try:
        db.session.delete(log)
        db.session.commit()
        flash('Study log deleted.', 'info')
    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting log: {str(e)}', 'danger')

    return redirect(url_for('user_study'))


@app.route('/user/fitness', methods=['GET', 'POST'])
@login_required
def user_fitness():
    if request.method == 'POST':
        date = datetime.strptime(request.form['date'], '%Y-%m-%d')
        activity = request.form['activity']
        duration = int(request.form['duration'])
        calories = int(request.form['calories'])

        log = FitnessLog(
            user_id=current_user.id,
            date=date,
            activity=activity,
            duration_min=duration,
            calories_burned=calories,
        )
        db.session.add(log)
        db.session.commit()
        flash('Fitness activity logged.')
        return redirect(url_for('user_fitness'))

    logs = current_user.fitness_logs

    # --- Basic metrics ---
    total_min = sum(l.duration_min for l in logs)
    total_cal = sum((l.calories_burned or 0) for l in logs)
    avg_min_per_day = total_min / max(
        1,
        (
            datetime.now().date()
            - (logs[0].date if logs else datetime.now().date())
        ).days,
    )

    # --- Weekly progress (current week: Monday–Sunday) ---
    today = datetime.now().date()
    start_of_week = today - timedelta(days=today.weekday())  # Monday
    weekly_logs = [l for l in logs if l.date >= start_of_week]
    weekly_min = sum(l.duration_min for l in weekly_logs)
    week_goal = 150  # minutes per week
    week_progress = min(100, (weekly_min / week_goal) * 100)

    # --- Daily trend (last 14 days) ---
    daily_totals = defaultdict(int)
    for log in logs:
        daily_totals[log.date.strftime('%Y-%m-%d')] += log.duration_min

    # Get last 14 days (including today)
    dates = [
        (today - timedelta(days=i)).strftime('%Y-%m-%d')
        for i in range(13, -1, -1)
    ]
    daily_data = [daily_totals.get(d, 0) for d in dates]

    # --- Activity breakdown ---
    activity_totals = defaultdict(int)
    for log in logs:
        activity_totals[log.activity] += log.duration_min

    activities = list(activity_totals.keys())
    activity_minutes = list(activity_totals.values())

    # --- Streak: consecutive days with at least 10 min of activity ---
    streak = 0
    if logs:
        study_days_set = sorted(set(log.date for log in logs), reverse=True)
        for i in range(30):  # max 30 days streak
            check_date = today - timedelta(days=i)
            # Check if there is any activity on this day (>=10 min)
            day_total = sum(
                l.duration_min for l in logs if l.date == check_date
            )
            if day_total >= 10:
                streak += 1
            else:
                break

    # --- Recent logs (last 10, newest first) ---
    recent_logs = logs[-10:][::-1]

    return render_template(
        'user/fitness.html',
        user=current_user,
        logs=recent_logs,
        total_min=total_min,
        total_cal=total_cal,
        avg_min_per_day=round(avg_min_per_day, 1),
        weekly_min=weekly_min,
        week_progress=round(week_progress, 1),
        dates=dates,
        daily_data=daily_data,
        activities=activities,
        activity_minutes=activity_minutes,
        streak=streak,
    )


@app.route('/fitness/edit/<int:log_id>', methods=['POST'])
@login_required
def edit_fitness(log_id):
    log = FitnessLog.query.get_or_404(log_id)
    if log.user_id != current_user.id:
        flash('You are not authorised to edit this activity.')
        return redirect(url_for('user_fitness'))

    date = datetime.strptime(request.form['date'], '%Y-%m-%d')
    activity = request.form['activity']
    duration = int(request.form['duration'])
    calories = int(request.form['calories'])

    log.date = date
    log.activity = activity
    log.duration_min = duration
    log.calories_burned = calories
    db.session.commit()
    flash('Activity updated successfully! ✅')
    return redirect(url_for('user_fitness'))


@app.route('/fitness/delete/<int:log_id>', methods=['POST'])
@login_required
def delete_fitness(log_id):
    log = FitnessLog.query.get_or_404(log_id)
    if log.user_id != current_user.id:
        flash('You are not authorised to delete this activity.')
        return redirect(url_for('user_fitness'))

    db.session.delete(log)
    db.session.commit()
    flash('Activity deleted successfully!')
    return redirect(url_for('user_fitness'))


# ------------------------- Additional pages needed by base.html -------------------------
@app.route('/privacy')
def privacy():
    return render_template('info.html', page_title='Privacy Policy', page_text='This demonstration stores account details and the activity you choose to enter in a local SQLite database. Do not enter sensitive personal information. Configure a real privacy policy before public deployment.')

@app.route('/terms')
def terms():
    return render_template('info.html', page_title='Terms and Conditions', page_text='This is a demonstration application. Its forecasts and simulations are illustrative only and are not professional financial, medical, or academic advice.')

@app.route('/contact')
def contact():
    return render_template('info.html', page_title='Contact', page_text='Contact details have not been provided for this project. Add your mentor or project team contact information before deployment.')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/admin')
@login_required
def admin_dashboard():
    if not current_user.is_admin:
        flash('Administrator access required.', 'warning')
        return redirect(url_for('user_dashboard'))
    return render_template('admin/dashboard.html', user_count=User.query.count())

@app.route('/admin/users')
@login_required
def admin_users():
    if not current_user.is_admin:
        flash('Administrator access required.', 'warning')
        return redirect(url_for('user_dashboard'))
    return render_template('admin/users.html', users=User.query.order_by(User.id.asc()).all())

@app.route('/user/simulation', methods=['GET', 'POST'])
@login_required
def user_simulation():
    result = None
    if request.method == 'POST':
        try:
            action = request.form.get('scenario', 'save')
            if action not in {'save', 'invest', 'spend', 'study_more', 'exercise'}:
                raise ValueError('Select a valid scenario.')
            amount = float(request.form.get('amount', '0'))
            if amount <= 0 or amount > 1_000_000_000:
                raise ValueError('Enter an amount greater than zero and within the allowed range.')
            result = simulate_scenario(current_user, action, amount)
            projected = current_user.savings + amount * 12 if action == 'save' else (amount * 12 * 1.1 if action == 'invest' else current_user.savings - amount * 12 if action == 'spend' else None)
            history = SimulationHistory(user_id=current_user.id, scenario=action, amount=amount,
                result_desc=result['desc'][:200], result_impact=result['impact'][:200],
                result_risk=result['risk'][:20], result_rec=result['rec'][:200],
                projected=projected, current=current_user.savings)
            db.session.add(history)
            db.session.commit()
        except (ValueError, TypeError) as exc:
            db.session.rollback()
            flash(str(exc) or 'Please enter a valid amount.', 'warning')
    history_items = SimulationHistory.query.filter_by(user_id=current_user.id).order_by(SimulationHistory.timestamp.desc()).limit(10).all()
    return render_template('user/simulation.html', result=result, history=history_items)

@app.route('/user/assistant', methods=['GET', 'POST'])
@login_required
def user_assistant():
    question = ''
    answer = ''
    if request.method == 'POST':
        question = request.form.get('question', '').strip()
        if not question:
            flash('Please enter a question.', 'warning')
        elif len(question) > 2000:
            flash('Please keep your question under 2,000 characters.', 'warning')
        else:
            if gemini_model:
                try:
                    response = gemini_model.generate_content(question)
                    answer = (getattr(response, 'text', '') or '').strip()
                except Exception:
                    answer = ''
            if not answer:
                lower = question.lower()
                if any(w in lower for w in ('study', 'exam', 'learn', 'gpa')):
                    answer = 'Try a focused 25–50 minute study block, define one measurable goal, remove distractions, and review what you learned at the end.'
                elif any(w in lower for w in ('fitness', 'exercise', 'workout', 'health')):
                    answer = 'Start with a realistic routine, increase activity gradually, and allow time for recovery. Choose activity suitable for your ability and circumstances.'
                elif any(w in lower for w in ('money', 'finance', 'saving', 'budget', 'expense')):
                    answer = 'Track income and expenses, separate needs from wants, set a small regular savings target, and review your budget weekly. This is general information, not financial advice.'
                else:
                    answer = 'Break the problem into a small next action, decide how you will measure progress, and review the result. Add a GEMINI_API_KEY to enable AI-generated responses.'
            db.session.add(ChatMessage(user_id=current_user.id, question=question, response=answer))
            db.session.commit()
    messages = ChatMessage.query.filter_by(user_id=current_user.id).order_by(ChatMessage.timestamp.desc()).limit(10).all()
    return render_template('user/assistant.html', question=question, answer=answer, messages=messages)

@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        occupation = request.form.get('occupation', '').strip() or 'Student'
        try:
            age = int(request.form.get('age', current_user.age or 25))
            if not username or not email or not (13 <= age <= 120):
                raise ValueError
            if User.query.filter(User.username == username, User.id != current_user.id).first():
                flash('That username is already in use.', 'warning')
                return redirect(url_for('settings'))
            if User.query.filter(User.email == email, User.id != current_user.id).first():
                flash('That email is already in use.', 'warning')
                return redirect(url_for('settings'))
            current_user.username, current_user.email = username, email
            current_user.occupation, current_user.age = occupation, age
            db.session.commit()
            flash('Settings saved.', 'success')
            return redirect(url_for('settings'))
        except (ValueError, TypeError):
            db.session.rollback()
            flash('Enter a valid username, email, and age between 13 and 120.', 'warning')
    return render_template('settings.html')



if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        create_default_admin()
    app.run(debug=True)
