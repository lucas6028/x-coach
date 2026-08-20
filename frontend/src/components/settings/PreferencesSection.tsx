import { useI18n } from "../../lib/i18n";
import LanguageSelect from "./LanguageSelect";
import { PaneRows, PaneTitle, SettingRow } from "./parts";

// Language. This is now the ONLY place to change it anywhere in the app — the chrome used to
// carry a duplicate picker in the top row, and the appearance row beside it is gone with the
// theme system. Which is why it sits in the pane that opens by default rather than behind a
// category, and why the LINE app (no navbar at all — see components/LiffAppShell) is no worse off.
export default function PreferencesSection() {
  const { t } = useI18n();
  return (
    <section>
      <PaneTitle>{t("settings.preferences")}</PaneTitle>
      <PaneRows>
        <SettingRow label={t("settings.language")}>
          <LanguageSelect />
        </SettingRow>
      </PaneRows>
    </section>
  );
}
