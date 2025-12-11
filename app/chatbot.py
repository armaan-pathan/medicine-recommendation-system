from flask import Flask, request, render_template, session
from model import diagnose_disease, generate_medication_response, generate_precaution_response, is_affirmative
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)
app.secret_key = 'your-secret-key'  # Replace with a secure key for production

@app.route('/', methods=['GET', 'POST'])
def chat():
    if 'chat_history' not in session or 'state' not in session:
        session['chat_history'] = [
            {"sender": "bot", "message": "Welcome to Doctor AI! Please describe your symptoms (e.g., I have a fever and cough). Note: This is not a substitute for professional medical advice. Consult a doctor."}
        ]
        session['state'] = 'awaiting_symptoms'
        session['diagnosis'] = None
        session.modified = True

    if request.method == 'POST':
        user_input = request.form.get('user_input', '').strip()
        chat_history = session['chat_history']
        state = session['state']

        if user_input.lower() == 'exit':
            session.clear()
            session['chat_history'] = [{"sender": "bot", "message": "Goodbye!"}]
            session['state'] = 'awaiting_symptoms'
            session['diagnosis'] = None
            session.modified = True
            return render_template('chat.html', chat_history=session['chat_history'])

        if state == 'awaiting_symptoms':
            chat_history.append({"sender": "user", "message": user_input})
            if is_affirmative(user_input):
                chat_history.append({"sender": "bot", "message": "It looks like you said 'yes' or similar, but I need your symptoms first. Please describe how you're feeling (e.g., fever, cough)."})
            else:
                diagnosis, error = diagnose_disease(user_input)
                if error:
                    chat_history.append({"sender": "bot", "message": error})
                elif not diagnosis:
                    chat_history.append({"sender": "bot", "message": "I couldn't diagnose a disease based on those symptoms. Try being more specific."})
                else:
                    chat_history.append({"sender": "bot", "message": f"Based on your symptoms, you may have {diagnosis['Disease']}. {diagnosis['Description']}"})
                    chat_history.append({"sender": "bot", "message": f"Would you like some advice on medications and precautions for {diagnosis['Disease']}? (yes/no)"})
                    session['diagnosis'] = diagnosis
                    session['state'] = 'awaiting_response'
        elif state == 'awaiting_response':
            chat_history.append({"sender": "user", "message": user_input})
            if is_affirmative(user_input):
                diagnosis = session['diagnosis']
                chat_history.append({"sender": "bot", "message": "Here’s some advice to help you feel better:"})
                chat_history.append({"sender": "bot", "message": f"- {generate_medication_response(diagnosis['Medications'])}"})
                chat_history.append({"sender": "bot", "message": f"- {generate_precaution_response(diagnosis['Precautions'])}"})
                chat_history.append({"sender": "bot", "message": "Please consult a doctor for a proper diagnosis and treatment plan."})
            else:
                chat_history.append({"sender": "bot", "message": "Okay, let me know if you have more symptoms or questions!"})
            session['state'] = 'awaiting_symptoms'
            session['diagnosis'] = None

        session['chat_history'] = chat_history
        session.modified = True

    return render_template('chat.html', chat_history=session['chat_history'])

@app.route('/reset', methods=['POST'])
def reset():
    session.clear()
    session['chat_history'] = [
        {"sender": "bot", "message": "Welcome to Doctor AI! Please describe your symptoms (e.g., I have a fever and cough). Note: This is not a substitute for professional medical advice. Consult a doctor."}
    ]
    session['state'] = 'awaiting_symptoms'
    session['diagnosis'] = None
    session.modified = True
    return render_template('chat.html', chat_history=session['chat_history'])

if __name__ == "__main__":
    app.run(debug=True, port=5001)   
