import { localEdgeLlm } from 'edge-intelligence-sdk';
import { Text, View } from 'react-native';

// This release smoke fixture intentionally constructs a session during module
// evaluation, so the success UI renders only after the native bridge works. An
// empty path selects the deterministic test model instead of loading a GGUF.
localEdgeLlm('');

export default function App() {
  return (
    <View>
      <Text>Edge Intelligence native bridge loaded.</Text>
    </View>
  );
}
