def read_result(prediction):
    """
    Function to interpret the prediction and return the corresponding response.
    Takes a prediction as input and returns a string response.
    """
    if int(prediction[0]) == 0:
        response="Bénin"
        return response
    elif int(prediction[0]) == 1:
        response="Scam"
        return response
    elif int(prediction[0]) == 2:
        response="Phishing"
        return response
    elif int(prediction[0]) == 3:
        response="Malware"
        return response