/* eslint-disable prettier/prettier */
/* eslint-disable react/react-in-jsx-scope */

import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import ModelStl from './components/ModelStl';
import Picker from './components/Picker';
import CrackImagePicker from './components/CrackImagePicker';
import TabsScreen from './components/TabsScreen';




const Stack = createNativeStackNavigator();

export default function StackNavigator() {
  return (
    <NavigationContainer>
      <Stack.Navigator
        screenOptions={{
          headerShown: false,
        }}
      >
                <Stack.Screen name="Tabs" component={TabsScreen} />

        <Stack.Screen name="Model" component={ModelStl} />




    </Stack.Navigator>
    </NavigationContainer>
  );
}
