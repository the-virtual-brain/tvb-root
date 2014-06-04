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
import org.openqa.selenium.WebElement;

import java.util.List;


/**
 * @author: ionel.ortelecan
 */
public class UserTest extends AbstractBaseTest {

    public UserTest() {
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


    public void testValidUserLogin() throws Exception {
        loginAdmin();
        assertEquals("The title of the page is not correct.", "The Virtual Brain. | User Profile", webDriver.getTitle());
        WebElement message = findElement(By.id("messageDiv"));
        boolean isMsgCorrect = "Your current working project is: Default_Project_admin".equals(message.getText()) ||
                "Welcome admin".equals(message.getText());
        assertTrue("The welcome message is not correct.", isMsgCorrect);

        //if the element is not found, NoSuchElementException is thrown
        WebElement logout = findElement(By.name("logout"));
        assertEquals("button", logout.getTagName());
        assertEquals("submit", logout.getAttribute("type"));
    }


    public void testInvalidUserLogin() throws Exception {
        login("lola", ADMIN_PASSWORD);
        assertEquals("The title of the page is not correct.", "The Virtual Brain. | Login", webDriver.getTitle());
        WebElement message = findElement(By.id("messageDiv"));
        assertEquals("The welcome message is not correct.", "Wrong username/password, or user not yet validated...", message.getText());

        //make sure that the username and password inputs are there
        findElement(By.id("username"));
        findElement(By.id("password"));

        login(ADMIN_USERNAME, "lola");
        assertEquals("The title of the page is not correct.", "The Virtual Brain. | Login", webDriver.getTitle());
        message = findElement(By.id("messageDiv"));
        assertEquals("The welcome message is not correct.", "Wrong username/password, or user not yet validated...", message.getText());

        //make sure that the username and password inputs are still there
        findElement(By.id("username"));
        findElement(By.id("password"));
    }


    public void testEmptyUsernameLogin() throws Exception {
        login("", ADMIN_PASSWORD);
        assertEquals("The title of the page is not correct.", "The Virtual Brain. | Login", webDriver.getTitle());
        WebElement message = findElement(By.id("messageDiv"));
        assertEquals("The welcome message is not correct.", "", message.getText());

        //make sure that the username and password inputs are still there
        findElement(By.id("username"));
        findElement(By.id("password"));

        //we do not check the error message because it is internationalized
        List<WebElement> errorMessages = webDriver.findElements(By.className("errorMessage"));
        assertEquals("There should be only one error msg.", 1, errorMessages.size());
        assertNotNull("The text of the error message should not be null", errorMessages.get(0).getText());
        assertTrue("The error message should not be empty", errorMessages.get(0).getText().length() > 0);
    }


    public void testNavigateToDataStructure() {
        loginAdmin();
        WebElement projectLink = webDriver.findElement(By.id("nav-project")).findElement(By.linkText("Project"));
        projectLink.click();

        if (isElementPresent(By.className("current-project-div"))) {
            WebElement dataStructureLink = webDriver.findElement(By.linkText("Data structure"));
            dataStructureLink.click();
        } else {
            WebElement dataStructureLink = webDriver.findElement(By.id("nav-project-data")).findElement(By.tagName("a"));
            dataStructureLink.click();
        }
        assertTrue("The canvas for the graph is missing.", isElementPresent(By.id("workflowCanvasDiv")));
        assertTrue("The canvas for the tree is missing.", isElementPresent(By.id("tree4")));

        logout();
    }

    //todo-io: add logout tests
}
