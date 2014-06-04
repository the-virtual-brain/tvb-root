/**
 * (c)  Baycrest Centre for Geriatric Care ("Baycrest"), 2012, all rights reserved.
 *
 * No redistribution or commercial re-sale is permitted.
 * Neither the name of Baycrest nor the names of its contributors may be used to endorse or promote
 * products or services derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY BAYCREST ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
 * NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL BAYCREST BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
 * OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.”
 *
 **/

package org.tvb.tests;

import org.junit.After;
import org.junit.Before;
import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.support.ui.ExpectedCondition;
import org.openqa.selenium.support.ui.WebDriverWait;

import java.util.List;
import java.util.UUID;


/**
 * @author: ionel.ortelecan
 */
public class BurstTest extends AbstractBaseTest {
    public String BURST_ACTIVE = "burst-active";
    public String BURST_STARTED = "burst-started";
    public String BURST_FINISHED = "burst-finished";
    public String BURST_ERROR = "burst-error";


    public BurstTest() {
        super();
    }
    
    @Before
    public void setUp() throws Exception {
        super.setUp();
    }

    @After
    public void tearDown() throws Exception {
        super.tearDown();
    }


    public void testRunBurstWithDefaultValues() throws Exception {
        createBurstWithDefaultValues();
    }


    public void testRunBurstWithDefaultValuesAndCleanAfter() throws Exception {
        String burstEntryId = createBurstWithDefaultValues();
        //TODO-io: fix me - commented because HtmlUnitDriver seems to have problems with btn click (edit btn)
//        removeBurst(burstEntryId);
    }

    //Note: tests that run sim. with ranges are commented because when running the tests using HtmlUnit driver
    //the steps set from tests, for those ranges, are not taken into account => a lot of operations are started for each test
    //This method is not working

//    public void testParameterSpaceExploration() throws Exception {
//        String burstNameStr = startBurstWithRange();
//        //check the burst statuses
//        String selectorScript = "var link = $(\"a:contains('" + burstNameStr + "')\"); " +
//                "var parent = link.parent()[0]; return parent;";
//        WebElement liItem = findElementByScript(selectorScript);
//        final String burstEntryId = liItem.getAttribute("id");
//        //wait 10 minutes for BURST_STARTED class to appear on element
//        waitForClassToAppearOnElement(burstEntryId, BURST_STARTED, 600, 2000);
//        //wait max 1 hour for the burst to finish
//        waitForClassToAppearOnElement(burstEntryId, BURST_FINISHED, 3600, 2000);
//
//        //check if the parameter space component is displayed
//        WebElement paramSpaceElem = findElement(By.id("section-pse"));
//        assertNotNull(paramSpaceElem);
//        assertEquals("The tag name of the parameter space component is not correct.",
//                        "section", paramSpaceElem.getTagName());
//        String styleAttr = paramSpaceElem.getAttribute("style");
//        assertNotNull(styleAttr);
//        assertTrue("The parameter space exploration component is expected to have the " +
//                        "display attribute set to block.", styleAttr.contains("block"));
//    }


    public void testLoadBurst() throws Exception {
        //todo: fix - find another way to make the driver to wait for the ajaxCall to finish
        navigateToBurstPage();
        selectNewBurstEntry();

        WebElement simulation_length = findElement(By.id("simulation_length"));
        String defaultSimLength = simulation_length.getAttribute("value");

        String firstSimLength = "5.0";
        String secondSimLength = "8.0";
        //launch first burst
        simulation_length = findElement(By.id("simulation_length"));
        simulation_length.clear();
        simulation_length.sendKeys(firstSimLength);
        String firstBurstName = "Burst_" + UUID.randomUUID().toString();
        WebElement burstNameElem = findElement(By.id("input-burst-name-id"));
        burstNameElem.sendKeys(firstBurstName);
        findElement(By.id("button-launch-new-burst")).click();
        Thread.sleep(timeToWaitForBurstToLoad * 1000);
        selectNewBurstEntry();

        //launch second burst
        simulation_length = findElement(By.id("simulation_length"));
        simulation_length.clear();
        simulation_length.sendKeys(secondSimLength);
        String secondBurstName = "Burst_" + UUID.randomUUID().toString();
        burstNameElem = findElement(By.id("input-burst-name-id"));
        burstNameElem.clear();
        burstNameElem.sendKeys(secondBurstName);
        findElement(By.id("button-launch-new-burst")).click();
        Thread.sleep(timeToWaitForBurstToLoad * 1000);
        selectNewBurstEntry();

        //check default burst
        simulation_length = findElement(By.id("simulation_length"));
        assertEquals("The default simulation length is not correct or the burst " +
                "couldn't be loaded before expring the time set for the Thread.sleep() method.",
                defaultSimLength, simulation_length.getAttribute("value"));
        //check first burst
        String selectorScript = "var link = $(\"a:contains('" + firstBurstName + "')\"); return link[0];";
        WebElement burstEntry = findElementByScript(selectorScript);
        burstEntry.click();

        Thread.sleep(timeToWaitForBurstToLoad * 1000);
        simulation_length = findElement(By.id("simulation_length"));
        assertEquals("The simulation length for the first burst is not corrector or the burst " +
                "couldn't be loaded before expring the time set for the Thread.sleep() method.",
                firstSimLength, simulation_length.getAttribute("value"));
        //check second burst
        selectorScript = "var link = $(\"a:contains('" + secondBurstName + "')\"); return link[0];";
        burstEntry = findElementByScript(selectorScript);
        burstEntry.click();

        Thread.sleep(timeToWaitForBurstToLoad * 1000);
        simulation_length = findElement(By.id("simulation_length"));
        assertEquals("The simulation length for the second burst is not correct or the burst " +
                "couldn't be loaded before expring the time set for the Thread.sleep() method.",
                secondSimLength, simulation_length.getAttribute("value"));
    }


    //Note: tests that run sim. with ranges are commented because when running the tests using HtmlUnit driver
    //the steps set from tests, for those ranges, are not taken into account => a lot of operations are started for each test
    //This method is not working

//    public void testLoadBurstWithRange() throws Exception {
//        String burstName = startBurstWithRange();
//        findElement(By.linkText("New Burst")).click();
//        Thread.sleep(timeToWaitForBurstToLoad * 1000);
//        findElement(By.linkText("New Burst")).click();
//        Thread.sleep(timeToWaitForBurstToLoad * 1000);
//        String paramSpaceScript = "return $(\".parameter-space-exploration[style*='display: none;']\")[0];";
//        WebElement paramSpaceElem = findElementByScript(paramSpaceScript);
//        assertNotNull(paramSpaceElem);
//        String selectPortletsComponentScript = "return $(\".view-portlets[style*='display: block;']\")[0];";
//        WebElement portletsContainerElem = findElementByScript(selectPortletsComponentScript);
//        assertNotNull(portletsContainerElem);
//        //load the burst
//        String selectorScript = "var link = $(\"a:contains('" + burstName + "')\"); return link[0];";
//        WebElement burstEntry = findElementByScript(selectorScript);
//        burstEntry.click();
//        Thread.sleep(timeToWaitForBurstToLoad * 1000);
//
//        paramSpaceScript = "return $(\".parameter-space-exploration[style*='display: block;']\")[0];";
//        paramSpaceElem = findElementByScript(paramSpaceScript);
//        assertNotNull(paramSpaceElem);
//        selectPortletsComponentScript = "return $(\".view-portlets[style*='display: none;']\")[0];";
//        portletsContainerElem = findElementByScript(selectPortletsComponentScript);
//        assertNotNull(portletsContainerElem);
//    }


    public void testCheckBurstNavigation() throws Exception {
        navigateToBurstPage();
        assertEquals("The title of the page is not correct.", "The Virtual Brain. | Simulation Cockpit", webDriver.getTitle());

        WebElement body = findElement(By.tagName("body"));
        assertEquals("nnn", "s-burst", body.getAttribute("id"));

        WebElement headerGroup = findElement(By.tagName("hgroup"));
        String h1Text = headerGroup.findElement(By.tagName("h1")).getText();
        assertEquals("The title of the page is not correct.", "Simulator", h1Text);
        String h2Text = headerGroup.findElement(By.tagName("h2")).getText();
        assertEquals("The title of the page is not correct.", "Simulation Cockpit", h2Text);
    }


    /**
     * Method used for launching a burst with range.
     * 
     * @return the burst name
     * @throws Exception if any exception appears
     */
    private String startBurstWithRange() throws Exception {
        navigateToBurstPage();
        selectNewBurstEntry();

        String burstNameStr = "Burst_" + UUID.randomUUID().toString();
        WebElement burstNameElem = findElement(By.id("input-burst-name-id"));
        burstNameElem.clear();
        burstNameElem.sendKeys(burstNameStr);

        WebElement select = findElement(By.id("model"));
        List<WebElement> allOptions = select.findElements(By.tagName("option"));
        for (WebElement option : allOptions) {
            if ("FitzHughNagumo".equals(option.getAttribute("value"))) {
                option.click();
                break;
            }
        }
        //open the range
        findElement(By.id("data_modelFitzHughNagumomodel_parameters_option_FitzHughNagumo_a_RANGER_buttonExpand")).click();
        //compute the step
        WebElement spinnerStepElem = findElement(By.id("data_modelFitzHughNagumomodel_parameters_option_FitzHughNagumo_a_RANGER_stepSpinner"));
        String min = spinnerStepElem.getAttribute("min");
        String max = spinnerStepElem.getAttribute("max");
        Float value = (Float.parseFloat(max) - Float.parseFloat(min)) / 3.0f;
        spinnerStepElem.clear();
        spinnerStepElem.sendKeys(value.toString());
        findElement(By.id("button-launch-new-burst")).click();
        
        return burstNameStr;
    }


    private void waitForClassToAppearOnElement(final String elemId, final String clsName, long timeOutInSeconds, long sleepInMillis) {
        WebDriverWait wait = new WebDriverWait(webDriver, timeOutInSeconds, sleepInMillis);
        wait.until(new ExpectedCondition<Boolean>() {
            public Boolean apply(WebDriver d) {
                if (!isElementPresent(By.id(elemId))) {
                    return false;
                }
                WebElement webElement = findElement(By.id(elemId));
                String classesAttr = webElement.getAttribute("class");
                if (classesAttr != null && classesAttr.trim().length() > 0) {
                    String[] allClasses = classesAttr.split(" ");
                    for (String cls : allClasses) {
                        if (clsName.equals(cls)) {
                            return true;
                        }

                        //todo: remove hardcoded class
                        if (BURST_ERROR.equals(cls)) {
                            fail("The burst failed to start.");
                        }
                    }
                }
                return false;
            }
        });
    }


    /**
     * Method used for navigating to burst.
     */
    private void navigateToBurstPage() {
        loginAdmin();
        WebElement navBurst = webDriver.findElement(By.id("nav-burst"));
        if ("Please select a working project first!".equals(navBurst.getAttribute("title"))) {
            WebElement projectLink = webDriver.findElement(By.id("nav-project")).findElement(By.linkText("Project"));
            projectLink.click();

            List<WebElement> projects = findElement(By.id("projectsForm")).
                    findElement(By.tagName("ul")).findElements(By.tagName("a"));
            boolean isProjectSelected = false;
            for (WebElement link : projects) {
                if ("Select this project to work with...".equals(link.getAttribute("title")) && "selector".equals(link.getAttribute("class"))) {
                    link.click();
                    isProjectSelected = true;
                    break;
                }
            }

            if (!isProjectSelected) {
                fail("You can't navigate to burst because no project could be selected.");
            }
        }

        //we search again for the 'nav-burst' element because it was removed from cache
        navBurst = webDriver.findElement(By.id("nav-burst"));
        WebElement burstLink = navBurst.findElement(By.linkText("Simulator"));
        burstLink.click();
    }

    /**
     * Creates a new burs entry.
     */
    private void selectNewBurstEntry() throws InterruptedException {
        boolean isAnyBurstSelected = checkIfThereIsAnyBurstSelected();
        if (isAnyBurstSelected) {
            WebElement newBurst = findElement(By.partialLinkText("New"));
            newBurst.click();
            Thread.sleep(timeToWaitForBurstToLoad * 1000);
            assertFalse("The new burst entry was not selected.", checkIfThereIsAnyBurstSelected());
        }
    }

    /**
     * @return <code>true</code> only if there is any burst selected
     */
    public boolean checkIfThereIsAnyBurstSelected() {
        WebElement burstHistory = findElement(By.id("burst-history"));
        List<WebElement> entries = burstHistory.findElements(By.tagName("li"));
        boolean isAnyBurstSelected = false;
        for (int i = 0; i < entries.size(); i++) {
            String allClasses = entries.get(i).getAttribute("class");
            if (allClasses != null && allClasses.contains(BURST_ACTIVE)) {
                isAnyBurstSelected = true;
                break;
            }
        }

        return isAnyBurstSelected;
    }

    /**
     * Runs a burst with its default values.
     *
     * @return returns the id of the created burst entry.
     * @throws Exception if test fails
     */
    private String createBurstWithDefaultValues() throws Exception {
        navigateToBurstPage();

        //make sure that the 'New Burst' entry is selected by default.
        selectNewBurstEntry();

        String burstName = "Burst_" + UUID.randomUUID().toString();
        WebElement burstNameElem = findElement(By.id("input-burst-name-id"));
        burstNameElem.clear();
        burstNameElem.sendKeys(burstName);
        findElement(By.id("button-launch-new-burst")).click();

        String selectorScript = "var link = $(\"a:contains('" + burstName + "')\"); var parent = link.parent()[0]; return parent;";
        WebElement liItem = findElementByScript(selectorScript);
        final String burstEntryId = liItem.getAttribute("id");
        //wait 10 minutes for BURST_STARTED class to appear on element
        waitForClassToAppearOnElement(burstEntryId, BURST_STARTED, maxTimeToWaitForBurstToStart, 2000);
        //wait max 1 hour for the burst to finish
        waitForClassToAppearOnElement(burstEntryId, BURST_FINISHED, maxTimeToWaitForBurstToFinish, 2000);
        //make sure the burst is active
        waitForClassToAppearOnElement(burstEntryId, BURST_ACTIVE, 1, 500);

        //todo: check if in the results tab were created some entries

        return burstEntryId;
    }


    /**
     * Removes a burst by a burst entry id.
     * When calling this method you should be on the burst page.
     *
     * @param burstEntryId the id of the li element in which is displayed the burst name
     * @throws Exception if the burst couldn't be removed
     */
    private void removeBurst(String burstEntryId) throws Exception {
        WebElement burstEntry = findElement(By.id(burstEntryId));
        WebElement editButton = burstEntry.findElement(By.className("action-edit"));
        if (editButton != null) {
            editButton.click();
        }
        WebElement removeButton = burstEntry.findElement(By.className("action-delete"));
        if (removeButton != null) {
            removeButton.click();
        }

        assertTrue("The burst couldn't be removed.", wasElementRemoved(By.id(burstEntryId)));
    }
}
