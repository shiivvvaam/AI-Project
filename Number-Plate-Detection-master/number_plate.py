import cv2

# Load the pre-trained Haar cascade classifier for number plates
harcascade="model/haarcascade_russian_plate_number.xml"

# Open the default camera
cap=cv2.VideoCapture(0)

# Set the camera resolution
cap.set(3,640)
cap.set(4,480)

# Set the minimum area of a number plate to be detected
min_area=500

# Initialize a counter for the number of detected number plates
count=0

# Start an infinite loop to continuously process video frames
while True:

    # Read a video frame
    success,img=cap.read()

    # Load the plate cascade classifier
    plate_cascade=cv2.CascadeClassifier(harcascade)

    # Convert the color image to grayscale
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Detect number plates in the grayscale image using the plate cascade classifier
    plates=plate_cascade.detectMultiScale(img_gray,1.1,4)

    # Iterate through the detected number plates
    for(x,y,w,h) in plates:

        # Calculate the area of the number plate
        area=w*h

        # Check if the area of the number plate is above the minimum area threshold
        if area>min_area:

            # Draw a bounding box around the number plate
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

            # Add a label to the number plate
            cv2.putText(img,"Number Plate",(x,y-5),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,0,255),2)

            # Extract the number plate image region of interest (ROI)
            img_roi=img[y: y+h , x:x+w]

            # Display the number plate ROI in a separate window
            cv2.imshow("ROI",img_roi)

    # Display the processed video frame with detected number plates
    cv2.imshow("Result",img)

    # Check for the 's' key to save the extracted number plate image
    count += 1
    if cv2.waitKey(1) & 0xFF == ord('s'):

        # Save the extracted number plate image to a file
        cv2.imwrite("plates/scanned_img_"+ str(count) +".jpg",img_roi)

        # Add a label to the processed video frame indicating that the number plate has been saved
        cv2.rectangle(img,(0,200),(640,300),(0,255,0),cv2.FILLED)
        cv2.putText(img,"PLATE SAVED",(150,265),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(0,0,255),2)

        # Display the processed video frame with the saved number plate label
        cv2.imshow("Results",img)

        # Wait for 500 milliseconds
        cv2.waitKey(500)
