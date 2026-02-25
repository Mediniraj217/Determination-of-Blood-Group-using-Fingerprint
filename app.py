from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session
import torch
from PIL import Image
import os
import mysql.connector
from werkzeug.security import generate_password_hash, check_password_hash
import torchvision.transforms as transforms
from io import BytesIO
from flask import send_file
from flask import render_template, make_response
from xhtml2pdf import pisa


app = Flask(__name__, template_folder='templates')
app.secret_key = 'your_secret_key'
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# MySQL Configuration
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'blood_db'
}

# Create uploads folder if not exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# MySQL connection function
def get_db_connection():
    return mysql.connector.connect(**db_config)

# CNN model definition
class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = torch.nn.Linear(64 * 56 * 56, 512)
        self.fc2 = torch.nn.Linear(512, 8)

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the trained model
model = SimpleCNN()
model.load_state_dict(torch.load('fingerprint_blood_group_model.pth', map_location=torch.device('cpu')))
model.eval()

# Image preprocessing
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    name = request.form['fullname']
    age = request.form['age']
    phone = request.form['phone']
    email = request.form['email']
    file = request.files['fingerprint']

    if file and file.filename != '':
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        img = Image.open(file_path).convert('RGB')
        img_tensor = data_transform(img).unsqueeze(0)

        with torch.no_grad():
            prediction = model(img_tensor)

        predicted_index = torch.argmax(prediction).item()
        blood_groups = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']
        predicted_group = blood_groups[predicted_index]

        # DB logic with debug print
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            sql = """INSERT INTO patient (fullname, age, phone, email, file_path, blood_g)
                     VALUES (%s, %s, %s, %s, %s, %s)"""
            values = (name, age, phone, email, file_path, predicted_group)
            cursor.execute(sql, values)
            conn.commit()
        except mysql.connector.Error as err:
            print("MySQL Error:", err)
            conn.rollback()
            return f"Database error occurred: {err}", 500
        finally:
            cursor.close()
            conn.close()

        return render_template('result.html', blood_group=predicted_group, patient_name=name)

    return "File missing!", 400



@app.route('/generate_pdf/<patient_name>', methods=['GET'])
def generate_pdf(patient_name):
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM patient WHERE fullname = %s ORDER BY id DESC LIMIT 1", (patient_name,))
        patient = cursor.fetchone()

        if not patient:
            return "Patient not found", 404

        # Render HTML template with patient data
        html = render_template('pdf_template.html', patient=patient)

        # Create PDF from HTML
        result = BytesIO()
        pdf = pisa.pisaDocument(BytesIO(html.encode("UTF-8")), result)

        if not pdf.err:
            response = make_response(result.getvalue())
            response.headers["Content-Type"] = "application/pdf"
            response.headers["Content-Disposition"] = f"attachment; filename={patient_name}_report.pdf"
            return response
        else:
            return "PDF generation failed", 500

    except Exception as e:
        print("PDF generation error:", e)
        return "Internal Server Error", 500

    finally:
        cursor.close()
        conn.close()



@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE email = %s', (email,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user[3], password):
            session['user_id'] = user[0]
            flash('Login successful!', 'success')
            return redirect(url_for('predict_blood_group'))
        else:
            flash('Invalid email or password!', 'error')
    return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        fullname = request.form['fullname']
        email = request.form['email']
        password = request.form['password']
        confirmpassword = request.form['confirmpassword']

        if password != confirmpassword:
            flash('Passwords do not match!', 'error')
            return redirect(url_for('signup'))

        hashed_password = generate_password_hash(password)

        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('INSERT INTO users (fullname, email, password) VALUES (%s, %s, %s)',
                           (fullname, email, hashed_password))
            conn.commit()
            flash('Account created successfully!', 'success')
            return redirect(url_for('login'))
        except mysql.connector.IntegrityError:
            flash('Email already exists!', 'error')
        finally:
            conn.close()
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('admin_id', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

@app.route('/predict_blood_group')
def predict_blood_group():
    if 'user_id' not in session:
        flash('Please log in to access this page.', 'error')
        return redirect(url_for('login'))
    return render_template('login2.html')

@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute('SELECT * FROM admin WHERE username = %s', (username,))
        admin = cursor.fetchone()
        conn.close()

        if admin and check_password_hash(admin['password'], password):
            session['admin_id'] = admin['id']
            flash('Admin login successful!', 'success')
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Invalid admin credentials!', 'error')
    return render_template('admin_login.html')

@app.route('/admin_dashboard')
def admin_dashboard():
    if 'admin_id' not in session:
        flash('Please log in as admin to continue.', 'error')
        return redirect(url_for('admin_login'))
    return render_template('admin_dashboard.html')

if __name__ == '__main__':
    app.run(debug=True)
