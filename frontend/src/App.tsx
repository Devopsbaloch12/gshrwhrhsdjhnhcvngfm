import { DashboardShell } from "./components/layout/DashboardShell";
import { VoiceAgentSection } from "./sections/VoiceAgentSection";
import { VoiceSection } from "./sections/VoiceSection";
import { ApiSection } from "./sections/ApiSection";
import { useActiveSection } from "./hooks/useActiveSection";

export default function App() {
  const { section, navigate } = useActiveSection();

  return (
    <DashboardShell active={section} onNavigate={navigate}>
      {section === "voice-agent" && <VoiceAgentSection onNavigate={navigate} />}
      {section === "voice" && <VoiceSection />}
      {section === "api" && <ApiSection />}
    </DashboardShell>
  );
}
