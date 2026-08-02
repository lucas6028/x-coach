import { useI18n } from "../../lib/i18n";
import LanguageSelect from "./LanguageSelect";
import ThemeSegmented from "./ThemeSegmented";
import { PaneRows, PaneTitle, SettingRow } from "./parts";

// Appearance and language. The web navbar carries the same two controls, but inside the LINE app
// there is no navbar (see components/LiffAppShell), so this section is the only place they exist
// there — which is why it sits in the pane that opens by default rather than behind a category.
export default function PreferencesSection() {
  const { t } = useI18n();
  return (
    <section>
      <PaneTitle>{t("settings.preferences")}</PaneTitle>
      <PaneRows>
        <SettingRow label={t("settings.appearance")}>
          <ThemeSegmented />
        </SettingRow>
        <SettingRow label={t("settings.language")}>
          <LanguageSelect />
        </SettingRow>
      </PaneRows>
    </section>
  );
}
