import flask
import pickle
import pandas as pd

with open(f'model/covid_19_svc_predictions.pkl', 'rb') as f:
        model = pickle.load(f)

app = flask.Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
def main():
        if flask.request.method == 'GET':
               return(flask.render_template('main.html'))
        if flask.request.method == 'POST':
               noFever = flask.request.form['noFever']
               yesFever = flask.request.form['yesFever']
               yesDiffultBreathing = flask.request.form['yesDifficultBreathing']
               noDifficultBreathing = flask.request.form['noDifficultBreathing']
               yesDryCough = flask.request.form['yesDryCough']
               noDryCough = flask.request.form['noDryCough']
               yesTiredness = flask.request.form['yesTiredness']
               noTiredness = flask.request.form['noTiredness']
               yesPains = flask.request.form['yesPains']
               noPains = flask.request.form['noPains']
               yesNasal = flask.request.form['yesNasal']
               noNasal = flask.request.form['noNasal']
               yesDiarrhea = flask.request.form['yesDiarrhea']
               noDiarrhea = flask.request.form['noDiarrhea']
               yesRunnyNose = flask.request.form['yesRunnyNose']
               noRunnyNose= flask.request.form['noRunnyNose']
               male = flask.request.form['male']
               female = flask.request.form['female']

               input_var = pd.DataFrame([[noFever, noDiarrhea, noDifficultBreathing, noDryCough, noNasal, noPains, noRunnyNose, noTiredness, male, female, yesDiarrhea, yesDiffultBreathing, yesDryCough, yesFever, yesNasal, yesRunnyNose, yesPains, yesTiredness]], columns = ['nofever', 'noDiarrhea', 'nodiffybreath', 'noDryCough', 'noNasal', 'noPains', 'noRunnyNose', 'noTiredness', 'male', 'female', 'yesDiarrhea', 'yesdiffybreath', 'yesDryCough', 'yesfever', 'yesNasal', 'yesRunnyNose', 'yesPains', 'yesTiredness'], dtype=float)

               prediction = model.predict(input_var)[0]

               return flask.render_template('main.html', original_input={'No fever': noFever, 'Yes fever': yesFever, 'No difficult breathing': noDifficultBreathing, 'Yes difficult breathing': yesDiffultBreathing, 'No dry cough': noDryCough, 'Yes dry cough': yesDryCough, 'No tiredness': noTiredness, 'Yes Tiredness': yesTiredness, 'No pains': noPains, 'Yes pains': yesPains, 'No nasal': noNasal, 'Yes nasal': yesNasal, 'No diarrhea': noDiarrhea, 'Yes diarrhea': yesDiarrhea, 'No runny nose': noRunnyNose, 'No fever': yesRunnyNose, 'Male': male, 'Female': female}, result = prediction,)


if __name__ == '__main__':
    app.run()