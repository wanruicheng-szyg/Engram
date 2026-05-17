import 'react-native-get-random-values'; // must be first — polyfills crypto.getRandomValues for Hermes
import { registerRootComponent } from 'expo';
import React, { useEffect, useState } from 'react';
import { StatusBar } from 'expo-status-bar';
import { ActivityIndicator, View } from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { SafeAreaProvider } from 'react-native-safe-area-context';

import { initCrypto } from './src/services/keystore';
import { loadKeypair, KeyPair } from './src/services/keystore';
import DashboardScreen   from './src/screens/DashboardScreen';
import StartMiningScreen from './src/screens/StartMiningScreen';
import WalletScreen      from './src/screens/WalletScreen';
import OnboardingScreen  from './src/screens/OnboardingScreen';
import TabBar            from './src/components/TabBar';
import { C } from './src/theme';

const Tab = createBottomTabNavigator();

function MainTabs() {
  return (
    <Tab.Navigator
      tabBar={props => <TabBar {...props} />}
      screenOptions={{ headerShown: false }}
    >
      <Tab.Screen name="Dashboard"   component={DashboardScreen} />
      <Tab.Screen name="Start Mining" component={StartMiningScreen} />
      <Tab.Screen name="Wallet"      component={WalletScreen} />
    </Tab.Navigator>
  );
}

function App() {
  const [ready,     setReady]     = useState(false);
  const [hasWallet, setHasWallet] = useState(false);

  useEffect(() => {
    (async () => {
      await initCrypto().catch(() => {});
      const kp = await loadKeypair();
      setHasWallet(kp !== null);
      setReady(true);
    })();
  }, []);

  if (!ready) {
    return (
      <View style={{ flex: 1, backgroundColor: C.bg, alignItems: 'center', justifyContent: 'center' }}>
        <StatusBar style="light" />
        <ActivityIndicator size="large" color={C.purple} />
      </View>
    );
  }

  return (
    <SafeAreaProvider>
      <StatusBar style="light" />
      <NavigationContainer>
        {hasWallet
          ? <MainTabs />
          : <OnboardingScreen onComplete={(_kp: KeyPair) => setHasWallet(true)} />
        }
      </NavigationContainer>
    </SafeAreaProvider>
  );
}

registerRootComponent(App);
