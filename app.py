from flask import Flask, jsonify, request
import kerasModel



app = Flask(__name__)

@app.route('/', methods=["GET"])
def root():
	return jsonify({"ok": True, "error": None, "key": "value"})

@app.route('/model', methods=["POST"])
def model():

	#fetch json body
	body = request.get_json()

	# if the json body isn't present, return an error
	if body is None or type(body) != dict:
		return jsonify({"ok": False, "error": "valid json response body is required"})

	# define the arguments
	params = ['usd_goal', 'term', 'category', 'blurb', 'subcategory']

	# make sure all the params are in the json body
	# for param in params:
	# 	if not body.get(param):
	# 		return jsonify({"ok": False, "error": "{} is a required parameter".format(param)})

	# here are the arguments now that you should insert into the model
	# and we know that they will always be there because of the checking we do two lines above
	## (body['usd_goal'], body['term'], body['category'], body['blurb'], body['subcategory'])
	resp = kerasModel.predict(body['term'], body['usd_goal'], body['category'], body['blurb'], body['subcategory'])

	return jsonify({"ok": True, "error": "None", "prediction": resp})


if __name__ == '__main__':
    app.run()



