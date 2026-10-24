import { expect, test } from "@playwright/test";

test.describe("research explorer", () => {
  test("loads the overview with strategies and receipts", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByTestId("honesty-strip")).toContainText(
      "RESEARCH ONLY",
    );
    const table = page.getByTestId("strategy-table");
    await expect(table).toBeVisible();
    // 2 equity-backed + 2 allocators + 5 sleeves from the real fixtures
    await expect(table.locator("tbody tr")).toHaveCount(9);
    await expect(
      page.getByTestId("receipt-summary-table").locator("tbody tr"),
    ).toHaveCount(396);
  });

  test("navigates to a strategy detail and renders charts", async ({
    page,
  }) => {
    await page.goto("/");
    await page
      .getByTestId("strategy-table")
      .getByRole("link", { name: "Carry champion", exact: true })
      .click();
    await expect(page).toHaveURL(/#\/strategy\/carry$/);
    await expect(page.getByTestId("strategy-page")).toBeVisible();
    await expect(page.getByTestId("historical-diagnostics")).not.toHaveAttribute("open");
    await expect(page.getByText("Model quality and promotion readiness are unmeasured here.", { exact: false })).toBeVisible();
    await page.getByTestId("historical-diagnostics").locator("summary").click();
    // equity + drawdown charts (svg role=img) with a line path
    const charts = page.getByTestId("line-chart");
    await expect(charts).toHaveCount(2);
    await expect(charts.first().locator("path.line")).toBeVisible();
    await expect(page.getByTestId("equity-summary")).toContainText(
      "daily bars",
    );
    await expect(page.getByTestId("stats-table")).toBeVisible();
  });

  test("receipt-stats strategy shows segments without an equity chart", async ({
    page,
  }) => {
    await page.goto("/#/strategy/amix_adaptive");
    await page.getByTestId("historical-diagnostics").locator("summary").click();
    await expect(page.getByTestId("no-equity-note")).toBeVisible();
    const stats = page.getByTestId("stats-table");
    await expect(stats).toBeVisible();
    await expect(stats).toContainText("development");
    await expect(stats).toContainText("historical_tail_exploratory");
    await expect(page.getByTestId("extras-table")).toContainText(
      "full_path_costs.commission",
    );
  });

  test("receipt list and verification panel render", async ({ page }) => {
    await page.goto("/#/receipts");
    const table = page.getByTestId("receipt-table");
    await expect(table.locator("tbody tr")).toHaveCount(396);
    await table
      .getByRole("link", { name: "corpus_epoch_d479e96760271bba" })
      .click();
    await expect(page).toHaveURL(
      /#\/receipt\/corpus_epoch_d479e96760271bba$/,
    );
    // verification surfaces: meta, client checks, digest table
    await expect(page.getByTestId("meta-panel")).toContainText(
      "corpus_epoch.v1",
    );
    const checks = page.getByTestId("checks-panel");
    await expect(checks).toContainText("research_only = true");
    await expect(checks).toContainText("live_pnl_claim = false");
    // WebCrypto digest of the fetched bytes is shown (64 hex chars)
    await expect(checks.locator(".check-detail").first()).toContainText(
      /^[0-9a-f]{64}$/,
    );
    const hashPanel = page.getByTestId("hash-panel");
    await expect(hashPanel).toContainText("epoch_root_sha256");
    // fast_replay embeds code_sha256 entries; public.py resolves at HEAD
    await page.goto("/#/receipt/fast_replay_p42_conformance_20260928");
    const fastHash = page.getByTestId("hash-panel");
    await expect(fastHash).toContainText("src/quant_fund/public.py");
    await expect(
      page.getByTestId("checks-panel"),
    ).toContainText("1/6 digests match a committed file");
  });

  test("hash routing handles direct loads and bad ids", async ({ page }) => {
    await page.goto("/#/receipt/verdict_real_drill");
    await expect(page.getByTestId("receipt-page")).toBeVisible();
    await page.goto("/#/strategy/does-not-exist");
    await expect(page.getByText("Unknown strategy")).toBeVisible();
    await page.goto("/#/bogus/path/here");
    await expect(page.getByTestId("overview-page")).toBeVisible();
    // malformed percent-escapes must not crash the router
    await page.goto("/#/receipt/%");
    await expect(page.getByTestId("overview-page")).toBeVisible();
  });

  test("no page errors on main navigation flows", async ({ page }) => {
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await page.goto("/");
    await page.goto("/#/strategy/carry_expanded");
    await page.goto("/#/receipts");
    await page.goto("/#/receipt/mcs_real_drill");
    await expect(page.getByTestId("receipt-page")).toBeVisible();
    expect(errors).toEqual([]);
  });
});
