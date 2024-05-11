import { StyleSheet, Text, View } from 'react-native'
import React from 'react'
import { WebView } from 'react-native-webview';


const ModelStl = () => {
  return (
    <WebView
      source={{ uri: 'http://192.168.29.190:5500/stl_viewer/sample.html' }}
      style={{ flex: 1 }}
      originWhitelist={['*']}
    />
  )
}

export default ModelStl

const styles = StyleSheet.create({})