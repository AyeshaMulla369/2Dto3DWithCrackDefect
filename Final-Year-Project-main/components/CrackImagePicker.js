/* eslint-disable prettier/prettier */
/* eslint-disable eol-last */
/* eslint-disable no-alert */
/* eslint-disable no-trailing-spaces */
/* eslint-disable no-unused-vars */
/* eslint-disable semi */
import { StyleSheet, Text, View , TouchableOpacity, Button, Image, ScrollView} from 'react-native'
import React, {useEffect, useState} from 'react'
import ImagePicker from 'react-native-image-crop-picker';
import axios from 'axios';
import Buffer from 'Buffer';
import CardComponent from './CardComponent';

const CrackImagePicker = () => {
    const [image, setImage] = useState(null);
    const [selectedImageUrl, setSelectedImageUrl] = useState(null);
    const [imageProcessed, setProcessedImageURI] = useState(null);
    const [imageRed, setRedImageURI] = useState(null);
    const [imagePred, setPredImageURI] = useState(null);

    const refresh = ()=>{
      setSelectedImageUrl(null);
      setProcessedImageURI(null);
      setRedImageURI(null);
      setPredImageURI(null);
    }



    const pickImage = async () => {
        try {
        const imagepick = await ImagePicker.openPicker({
            cropping: true,
          });
          console.log(imagepick);
          setImage(imagepick);
          const cloudUri = await handleUpload(imagepick.path);
          setSelectedImageUrl(cloudUri);
        } catch (error) {
          console.log('ImagePicker Error: ', error);
        }
    };

    const handleUpload = async (post) => {
        if (post != null) {
          const formData = new FormData();
          formData.append('file', {
            uri: post,
            type: 'image/jpeg',
            name: 'uploaded_image',
          });
          formData.append('upload_preset', 'yivau9kc'); // Use your actual upload preset name
      
          try {
            const response = await axios.post('https://api.cloudinary.com/v1_1/dnz4gywty/image/upload', formData, {
              headers: {
                'Content-Type': 'multipart/form-data',
              },
            });
      
            // Upon successful upload, Cloudinary typically responds with the image URL
            const imageUrl = response.data.secure_url;
            console.log(imageUrl);
      
            return imageUrl;
      
          } catch (error) {
            // console.error('Error uploading image:', error);
            // console.log('Cloudinary Response:', error.response.data);
            alert('Please select an image for post.');
          }
        } else {
          alert('Please select an image');
        }
      };

console.log(encodeURIComponent(selectedImageUrl))

      const uploadImage = async()=>{
          await getPred();
          await getProcessed();
          await getRed();
      }


      

      const getProcessed = async () => {
        if (!selectedImageUrl) {
          alert('Please select an image first!');
          return;
        }
    
        try {
            // Send a POST request to the histogram endpoint
        const response = await axios.post(`http://192.168.29.190:8000/processed_image/?url=${encodeURIComponent(selectedImageUrl)}`,
        {},  
        {
            headers: {
              'Content-Type': 'application/json',
            },
            responseType: 'blob',
          }
        );
        const blob = new Blob([response.data], { type: 'image/png' });
        const reader = new FileReader();
        reader.onloadend = () => {
          setProcessedImageURI(reader.result);
        };
        reader.readAsDataURL(blob);
        } catch (error) {
          console.error('Error uploading image:', error);
          alert('Error uploading image. Please try again later.');
        }
      };

      const getRed = async () => {
        if (!selectedImageUrl) {
          alert('Please select an image first!');
          return;
        }
    
        try {
            // Send a POST request to the histogram endpoint
        const response = await axios.post(`http://192.168.29.190:8000/red_parts/?url=${encodeURIComponent(selectedImageUrl)}`,
        {},  
        {
            headers: {
              'Content-Type': 'application/json',
            },
            responseType: 'blob',
          }
        );
        const blob = new Blob([response.data], { type: 'image/png' });
        const reader = new FileReader();
        reader.onloadend = () => {
          setRedImageURI(reader.result);
        };
        reader.readAsDataURL(blob);
        } catch (error) {
          console.error('Error uploading image:', error);
          alert('Error uploading image. Please try again later.');
        }
      };

      const getPred = async () => {
        if (!selectedImageUrl) {
          alert('Please select an image first!');
          return;
        }
    
        try {
            // Send a POST request to the histogram endpoint
        const response = await axios.post(`http://192.168.29.190:8000/predict_on_crops/?url=${encodeURIComponent(selectedImageUrl)}`,
        {},  
        {
            headers: {
              'Content-Type': 'application/json',
            },
            responseType: 'blob',
          }
        );

        console.log(response.data)
        const blob = new Blob([response.data], { type: 'image/png' });
        const reader = new FileReader();
        reader.onloadend = () => {
          setPredImageURI(reader.result);
        };
        reader.readAsDataURL(blob);
        } catch (error) {
          console.error('Error uploading image:', error);
          alert('Error uploading image. Please try again later.');
        }
      };

  
      return (
        <View style={styles.view}>
          <ScrollView contentContainerStyle={styles.container}>
            {selectedImageUrl ? 
              (
                <TouchableOpacity style={styles.button} onPress={refresh}>
                  <Text style={styles.buttonText}>Reset Image</Text>
                </TouchableOpacity>
              ) : (
                <TouchableOpacity style={styles.button} onPress={pickImage}>
                  <Text style={styles.buttonText}>Pick an image from camera roll</Text>
                </TouchableOpacity>
              )
            }
            {selectedImageUrl && <CardComponent imageUri={selectedImageUrl} title="Input Image" />}



            {imagePred &&  <View style={{
      backgroundColor: '#fff',
      borderRadius: 8,
      padding: 16,
      shadowColor: '#000',
      shadowOffset: {
        width: 0,
        height: 2,
      },
      shadowOpacity: 0.25,
      shadowRadius: 3.84,
      elevation: 5,
      alignItems: 'center',
      marginVertical: 10,
    }}>
      <Text style={{
        fontSize: 18,
        fontWeight: 'bold',
        marginBottom: 8,
      }}>Output Images</Text>
            {imagePred &&<View style={{alignItems:'center'}}>
              <Image source={{ uri: imagePred}} style={styles.image} />
              <Text>Zonal visualisation of crack</Text>
            </View> }
            {imageProcessed &&<View style={{alignItems:'center'}}>
              <Image source={{ uri: imageProcessed }} style={{...styles.image,width:200,height:200}} />
              <Text>Pixelated visualisation of crack</Text>
            </View> }
            {imageRed &&<View style={{alignItems:'center'}}>
              <Image source={{ uri: imageRed }} style={{...styles.image,width:200,height:200}} />
              <Text>Length Width visualisation of crack</Text>
            </View> }

            </View>}
            
    
            {selectedImageUrl && !imagePred && <TouchableOpacity style={styles.button} onPress={uploadImage}>
                  <Text style={styles.buttonText}>Analyze Crack</Text>
                </TouchableOpacity>}
    
          </ScrollView>
        </View>
      );
    };
    
    export default CrackImagePicker;
    
    const styles = StyleSheet.create({
      container: {
        alignItems: 'center',
        justifyContent: 'center',
      },
      view:{
        flex:1,
      },
      button: {
        backgroundColor: '#4169e1',
        borderColor: '#C0C0C0',
        borderWidth: 1,
        width: '80%',
        borderRadius: 10,
        padding: 12,
        alignItems: 'center',
        justifyContent: 'center',
        shadowColor: '#000',
        shadowOffset: {
          width: 0,
          height: 3,
        },
        shadowOpacity: 0.27,
        shadowRadius: 4.65,
        elevation: 6,
        marginVertical: 20,
      },
      buttonText: {
        color: 'white',
        fontSize: 16,
      },
      image: {
        width: 280,
        height: 250,
        marginTop: 20,
      },
    });